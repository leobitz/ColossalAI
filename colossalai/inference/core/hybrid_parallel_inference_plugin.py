import ctypes
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from types import MethodType
from typing import Any, Callable, Dict, Iterator, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, inf
from torch.distributed import ProcessGroup, get_world_size
from torch.nn import Module, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from colossalai.inference.utils import get_model_size
from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.checkpoint_io import CheckpointIO, HybridParallelCheckpointIO
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import AMPModelMixin, ModelWrapper, OptimizerWrapper
from colossalai.interface.optimizer import DistributedOptim
from colossalai.nn.optimizer import DistGaloreAwamW, cast_to_distributed
from colossalai.inference.core.one_f_one_b import OneForwardOneBackwardSchedule # InterleavedSchedule, 
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import GradientCheckpointConfig, ShardConfig, ShardFormer
from colossalai.shardformer.layer.utils import SeqParallelUtils
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.tensor.d_tensor.api import is_distributed_tensor
from colossalai.zero.low_level import LowLevelZeroOptimizer
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
import sys

SUPPORT_SP_MODE = ["split_gather", "ring", "all_to_all"]

PRECISION_TORCH_TYPE = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def _convert_floating_point(x, dtype: torch.dtype = torch.float16):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype)
    return x


class HybridParallelInferenceModule(ModelWrapper, AMPModelMixin):
    def __init__(
        self,
        module: Module,
        precision: str,
        shard_config: ShardConfig,
        dp_group: ProcessGroup,
        tp_group: ProcessGroup,
        sp_group: ProcessGroup,
        # use_ddp: bool,
        # ddp_config: dict,
        custom_policy: Policy,
    ) -> None:
        self.stage_manager = shard_config.pipeline_stage_manager
        self.shard_config = shard_config
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.sp_group = sp_group
        # self.use_dpp = use_ddp
        # self.require_grad_sync = False
        # print("======= custom_policy =========== ", custom_policy)
        # shard_config.pipeline_stage_manager = None
        torch.cuda.empty_cache()
        init_gpu_memory = torch.cuda.mem_get_info()[0]
        shardformer = ShardFormer(shard_config)
        if custom_policy is not None:
            assert isinstance(custom_policy, object)
        
        # memory_usage_bytes = sum(sys.getsizeof(param.storage()) for param in module.parameters())
        # memory_usage_gb = memory_usage_bytes / (1024 ** 3)
        # print(get_model_size(module), "GB---", type(module))
        module, self.shared_params = shardformer.optimize(module, policy=custom_policy)
        self.device = get_accelerator().get_current_device()
        # print("the device is ", self.device)
        # if self.verbose:
        #     self.logger.info(f"the device is {self.device}")
        wrapper  = ModelWrapper(module).to(self.device)
        # print(get_model_size(module), "GB---", type(wrapper))
        # setting process groups for shared parameters
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        # print("init gpu memory is ", init_gpu_memory / (1024 ** 3), "free gpu memory is ", free_gpu_memory / (1024 ** 3))
        peak_memory = init_gpu_memory - free_gpu_memory
        # print(f"Rank [{dist.get_rank()}], Model Weight Max Occupy {peak_memory / (1024 ** 3)} GB, Model size: {get_model_size(module)} GB")
        module = wrapper
        self.shared_param_process_groups = []
        for shared_param in self.shared_params:
            if len(shared_param) > 0:
                self.shared_param_process_groups.append(
                    self.stage_manager.init_process_group_by_stages(list(shared_param.keys()))
                )

        # setting mixed_precision
        self.mixed_precision = None
        if precision == "fp16":
            self.mixed_precision = torch.float16
        elif precision == "bf16":
            self.mixed_precision = torch.bfloat16
        if self.mixed_precision is not None:
            module = module.to(self.mixed_precision)
        module = module.to(get_accelerator().get_current_device())

        # setting input type cast when using mixed precision
        self.convert_fn = None
        if self.mixed_precision is not None:
            self.convert_fn = partial(_convert_floating_point, dtype=self.mixed_precision)

        # # setting ddp configs
        # if use_ddp:
        #     # convert model to sync bn
        #     module = SyncBatchNorm.convert_sync_batchnorm(module, dp_group)
        #     # wrap the model with PyTorch DDP
        #     module = DDP(module, process_group=dp_group, **ddp_config)

        super().__init__(module)

    def sync_shared_params(self):
        for shared_param, group in zip(self.shared_params, self.shared_param_process_groups):
            if self.stage_manager.stage in shared_param:
                param = shared_param[self.stage_manager.stage]
                dist.all_reduce(param.grad, group=group)
            dist.barrier()

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable automatic gradient synchronization (all-reduce) and allow manual synchronization
        when 'no_sync' is active. Alternatively, synchronization will occur in the first forward-backward pass
        when exiting the context.
        """

        # Store the current value of 'require_grad_sync' to restore it later.
        old_require_grad_sync = self.require_grad_sync
        # Disable automatic gradient synchronization.
        self.require_grad_sync = False
        try:
            if self.use_dpp:
                # If using data parallel processing (use_dpp), disable synchronization too.
                with self.module.no_sync():
                    yield
            else:
                yield
        finally:
            # Restore the original value of 'require_grad_sync'.
            self.require_grad_sync = old_require_grad_sync

    def sync_dp_grads(self):
        r"""
        Synchronize gradients across data parallelism (DP) if the DP group size is greater than 1.
        This function performs an all-reduce operation to combine gradients from different devices in the DP group.

        Args:
            None

        Returns:
            None
        """

        # Check if the DP group size is 1, meaning no synchronization is needed.
        if self.dp_group.size() == 1:
            return

        # Iterate through the model's parameters and perform gradient synchronization.
        for p in self.module.parameters():
            if p.grad is not None:
                # Perform all-reduce to combine gradients from different devices.
                dist.all_reduce(p.grad, group=self.dp_group)
                # Normalize the gradient by dividing it by the DP group size.
                p.grad.div_(self.dp_group.size())

    def sync_sp_grads(self, grads: Optional[List[torch.Tensor]] = None):
        r"""
        Synchronize gradients that are partially derived within sequence parallelism
        if sequence parallelism is enabled. Gradients can be provided explicitly or extracted
        from the module.

        Args:
            grads (Optional[List[torch.Tensor]]): A list of gradient tensors to synchronize. If not
                provided, gradients will be extracted from the model.

        Returns:
            None
        """

        if self.shard_config.enable_sequence_parallelism:
            if self.shard_config.sequence_parallelism_mode == "all_to_all":
                return

            if self.shard_config.sequence_parallelism_mode in ["split_gather", "ring"]:
                # If sequence parallelism is enabled and mode is split_gather or ring, gradients are synchronized
                # across the tensor parallelism group.
                group = self.tp_group
            else:
                raise ValueError(f"Unknown sequence parallelism mode: {self.shard_config.sequence_parallelism_mode}")

            if grads is not None:
                # Synchronize provided gradient tensors across the tensor parallelism group.
                SeqParallelUtils.allreduce_partial_data_grad(process_group=group, grads=grads)
            else:
                # Synchronize gradients from the model across the tensor parallelism group.
                SeqParallelUtils.allreduce_partial_data_grad(process_group=group, model=self.module)

    def forward(self, *args, **kwargs):
        if self.convert_fn is not None:
            args = tree_map(self.convert_fn, args)
            kwargs = tree_map(self.convert_fn, kwargs)
        return super().forward(*args, **kwargs)

    def unwrap(self):
        module = super().unwrap()
        if isinstance(module, DDP):
            module = module.module
        return module


def get_param_info(optim: Optimizer):
    # Get a backup of necessary information of parameters for future use, which includes:
    # 1. A complete param_group, with params in the form of param_id
    # 2. A mapping from param address (obtained using id(param)) to integer param_id
    # 3. A mapping from integer param_id to param address.
    # 4. A mapping from param_address (obtained using id(param)) to the original shape of parameter before sharding.
    # When Zero is used, the params here are fp16/bf16 model params rather than fp32 master params in optimizer.

    if optim is None:
        return {}
    param_info = {"param_groups": [], "param2id": {}, "id2param": {}, "param2shape": {}}
    start_index = 0
    for group in optim.param_groups:
        packed_group = {k: v for k, v in group.items() if k != "params"}
        packed_group["params"] = []

        for param_id, param in enumerate(group["params"], start_index):
            original_shape = param.shape if isinstance(param, torch.Tensor) else None
            packed_group["params"].append(param_id)
            param_info["param2id"][id(param)] = param_id
            param_info["id2param"][param_id] = id(param)
            param_info["param2shape"][id(param)] = original_shape

        param_info["param_groups"].append(packed_group)
        start_index += len(group["params"])

    return param_info


def init_pipeline_optimizer(optim: Optimizer, model: Module):
    model_params = set(model.parameters())
    new_param_groups = []
    for group in optim.param_groups:
        params = [p for p in group["params"] if p in model_params]
        new_param_groups.append({**group, "params": params})
    optim.__setstate__({"param_groups": new_param_groups})


class HybridParallelInferencePlugin(PipelinePluginBase):
    """
    Plugin for Hybrid Parallel Training.
    Tensor parallel, pipeline parallel and data parallel(DDP/ZeRO) can be picked and combined in this plugin.
    The size of tp and pp should be passed in by user, then the size of dp is automatically calculated from dp_size = world_size / (tp_size * pp_size).

    ```python
    from colossalai.booster import Booster
    from colossalai.booster.plugin import HybridParallelPlugin

    model, train_dataset, optimizer, criterion = ...
    plugin =  HybridParallelPlugin(tp_size=2, pp_size=2)

    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, train_dataloader, _ = booster.boost(model, optimizer, criterion, train_dataloader)
    ```

    Args:
        tp_size (int): The size of tensor parallelism. Tensor parallelism will not be used when tp_size is set to 1.
        pp_size (int): The number of pipeline stages in pipeline parallelism. Pipeline parallelism will not be used when pp_size is set to 1.
        sp_size (int): The size of sequence parallelism.
        precision (str, optional): Specifies the precision of parameters during training.
                                    Auto-mixied precision will be used when this argument is set to 'fp16' or 'bf16', otherwise model is trained with 'fp32'.
                                    Defaults to 'fp16'.
        zero_stage (int, optional): The stage of ZeRO for data parallelism. Can only be choosed from [0, 1, 2].
                                        When set to 0, ZeRO will not be used. Defaults to 0.
        enable_all_optimization (bool, optional): Whether to switch on all the optimizations supported by Shardformer.
                                                    Currently all the optimization methods include fused normalization, flash attention and JIT.
                                                    Defaults to False.
        enable_fused_normalization (bool, optional): Whether to switch on fused normalization in Shardformer. Defaults to False.
        enable_flash_attention (bool, optional): Whether to switch on flash attention in Shardformer. Defaults to False.
        enable_jit_fused (bool, optional): Whether to switch on JIT in Shardformer. Default to False.
        enable_sequence_parallelism (bool): Whether to turn on sequence parallelism in Shardformer. Defaults to False.
        sequence_parallelism_mode (str): The Sequence parallelism mode. Can only be choosed from ["split_gather", "ring", "all_to_all"]. Defaults to "split_gather".
        enable_sequence_overlap (bool): Whether to turn on sequence overlap in Shardformer. Defaults to False.
        parallel_output (bool): Whether to keep the output parallel when enabling tensor parallelism. Default to True.
        num_microbatches (int, optional): Number of microbatches when using pipeline parallelism. Defaults to None.
        microbatch_size (int, optional): Microbatch size when using pipeline parallelism.
            Either ``num_microbatches`` or ``microbatch_size`` should be provided if using pipeline.
            If ``num_microbatches`` is provided, this will be ignored. Defaults to None.
        initial_scale (float, optional): The initial loss scale of AMP. Defaults to 2**16.
        min_scale (float, optional): The minimum loss scale of AMP. Defaults to 1.
        growth_factor (float, optional): The multiplication factor for increasing loss scale when using AMP. Defaults to 2.
        backoff_factor (float, optional): The multiplication factor for decreasing loss scale when using AMP. Defaults to 0.5.
        growth_interval (int, optional): The number of steps to increase loss scale when no overflow occurs when using AMP. Defaults to 1000.
        hysteresis (int, optional):  The number of overflows before decreasing loss scale when using AMP. Defaults to 2.
        max_scale (float, optional): The maximum loss scale of AMP. Defaults to 2**32.
        max_norm (float, optional): Maximum norm for gradient clipping. Defaults to 0.
        broadcast_buffers (bool, optional): Whether to broadcast buffers in the beginning of training when using DDP. Defaults to True.
        ddp_bucket_cap_mb (int, optional): The bucket size in MB when using DDP. Defaults to 25.
        find_unused_parameters (bool, optional): Whether to find unused parameters when using DDP. Defaults to False.
        check_reduction (bool, optional): Whether to check reduction when using DDP. Defaults to False.
        gradient_as_bucket_view (bool, optional): Whether to use gradient as bucket view when using DDP. Defaults to False.
        static_graph (bool, optional): Whether to use static graph when using DDP. Defaults to False.
        zero_bucket_size_in_m (int, optional): Gradient reduce bucket size in million elements when using ZeRO. Defaults to 12.
        cpu_offload (bool, optional): Whether to open cpu_offload when using ZeRO. Defaults to False.
        communication_dtype (torch.dtype, optional): Communication dtype when using ZeRO. If not specified, the dtype of param will be used. Defaults to None.
        overlap_communication (bool, optional): Whether to overlap communication and computation when using ZeRO. Defaults to True.
        custom_policy (Policy, optional): Custom policy for Shardformer. Defaults to None.
        pp_style (str, optional): The style for pipeline parallelism. Defaults to '1f1b'.
        num_model_chunks (int, optional): The number of model chunks for interleaved pipeline parallelism. Defaults to 1.
        gradient_checkpoint_config (GradientCheckpointConfig, optional): Configuration for gradient checkpointing. Defaults to None.
        enable_metadata_cache (bool, optional): Whether to enable metadata cache for pipeline parallelism. Defaults to True.
        make_vocab_size_divisible_by (int, optional): it's used when padding the vocabulary size, to make it choose an faster kenel. Default to 64.

    """

    def __init__(
        self,
        tp_size: int,
        pp_size: int,
        sp_size: int = None,
        precision: str = "fp16",
        zero_stage: int = 0,
        enable_all_optimization: bool = False,
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        enable_sequence_parallelism: bool = False,
        sequence_parallelism_mode: str = None,
        enable_sequence_overlap: bool = False,
        parallel_output: bool = True,
        num_microbatches: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0,
        broadcast_buffers: bool = True,
        ddp_bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        zero_bucket_size_in_m: int = 12,
        cpu_offload: bool = False,
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = True,
        custom_policy: Policy = None,
        pp_style: str = "1f1b",
        num_model_chunks: int = 1,
        num_layers_per_stage: Optional[List[int]] = None,
        gradient_checkpoint_config: Optional[GradientCheckpointConfig] = None,
        enable_metadata_cache: bool = True,
        make_vocab_size_divisible_by: int = 64,
        dp_outside: bool = True,
        sharding_extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        assert (
            dist.get_world_size() % (tp_size * pp_size) == 0
        ), f"World size {dist.get_world_size()} is not divisible by tp_size {tp_size} * pp_size {pp_size}"

        if enable_sequence_parallelism:
            self.sequence_parallelism_mode = (
                sequence_parallelism_mode if sequence_parallelism_mode is not None else "all_to_all"
            )
            assert (
                self.sequence_parallelism_mode in SUPPORT_SP_MODE
            ), f"Sequence parallelism mode {self.sequence_parallelism_mode} is not in the supported list {SUPPORT_SP_MODE}"
            if self.sequence_parallelism_mode in ["split_gather", "ring"]:
                assert (
                    tp_size > 1
                ), f"Sequence parallelism mode {self.sequence_parallelism_mode} must be enabled when using tensor parallelism"
                if sp_size != 1:
                    warnings.warn(
                        f"The sp_size will be the same as tp_size in sequence parallelism mode {self.sequence_parallelism_mode}, will ignore the given sequence parallelism size."
                    )
                self.sp_size = 1
                self.dp_size = dist.get_world_size() // (tp_size * pp_size)
            elif self.sequence_parallelism_mode in ["all_to_all"]:
                self.sp_size = 1 if sp_size is None else sp_size
                self.dp_size = dist.get_world_size() // (self.sp_size * pp_size * tp_size)
        else:
            self.dp_size = dist.get_world_size() // (tp_size * pp_size)
            assert (
                sp_size == 1 or sp_size is None
            ), f"You should not set sp_size when sequence parallelism is not enabled."
            self.sp_size = 1

        self.tp_size = tp_size
        self.pp_size = pp_size
        self.precision = precision
        self.zero_stage = zero_stage
        self.cpu_offload = cpu_offload
        self.enable_all_optimization = enable_all_optimization
        self.enable_fused_normalization = enable_fused_normalization
        self.enable_flash_attention = enable_flash_attention
        self.enable_jit_fused = enable_jit_fused
        self.enable_sequence_parallelism = enable_sequence_parallelism
        # if dp_outside:
        #     (
        #         self.tp_axis,
        #         self.pp_axis,
        #         # self.dp_axis,
        #         # self.sp_axis,
        #     ) = (
        #         0,
        #         1,
        #         # 2,
        #         # 3,
        #     )
        #     self.pg_mesh = ProcessGroupMesh(self.pp_size, self.tp_size)
        # else:
        self.pp_axis, self.tp_axis, self.dp_axis, self.sp_axis = 0, 1, 2, 3
        self.pg_mesh = ProcessGroupMesh(self.pp_size, self.tp_size, self.dp_size, self.sp_size)

        self.stage_manager = None
        self.schedule: OneForwardOneBackwardSchedule = None
        self.custom_policy = custom_policy
        # assert zero_stage in (0, 1, 2)
        # if self.pp_size > 1:
        # assert pp_style in ["1f1b", "interleaved"], "Unsupported pipeline parallelism style"
        assert pp_style == "interleaved" or num_model_chunks == 1, "num_model_chunks must be 1 when using 1f1b"
        assert (
            num_microbatches is not None or microbatch_size is not None
        ), "num_microbatches or microbatch_size must be specified when using pipeline parallelism"
        # assert self.zero_stage <= 1, "zero stage must be 0 or 1 when using pipeline parallelism"
        self.stage_manager = PipelineStageManager(
            self.pg_mesh,
            pipeline_axis=self.pp_axis,
            enable_interleave=pp_style == "interleaved",
            num_model_chunks=num_model_chunks,
            num_layers_per_stage=num_layers_per_stage,
        )
        
        if pp_style == "interleaved":
            # assert num_model_chunks > 1, "number of model chunks must be > 1 when using interleaved"
            # self.schedule = InterleavedSchedule(
            #     stage_manager=self.stage_manager,
            #     num_model_chunks=num_model_chunks,
            #     num_microbatch=num_microbatches,
            #     microbatch_size=microbatch_size,
            #     enable_metadata_cache=enable_metadata_cache,
            # )
            pass
        elif pp_style == "1f1b":
            self.schedule = OneForwardOneBackwardSchedule(
                stage_manager=self.stage_manager,
                num_microbatches=num_microbatches,
                microbatch_size=microbatch_size,
                enable_metadata_cache=enable_metadata_cache,
            )
        else:
            raise NotImplementedError()

        self.tp_group = self.pg_mesh.get_group_along_axis(self.tp_axis)
        self.dp_group = self.pg_mesh.get_group_along_axis(self.dp_axis)
        self.pp_group = self.pg_mesh.get_group_along_axis(self.pp_axis)
        # if self.enable_sequence_parallelism and self.sequence_parallelism_mode in ["split_gather", "ring"]:
        #     self.sp_group = self.pg_mesh.get_group_along_axis(self.tp_axis)
        # else:
        self.sp_group = self.pg_mesh.get_group_along_axis(self.sp_axis)

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=self.tp_group,
            # sequence_parallel_process_group=self.sp_group,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=self.tp_size > 1,
            enable_all_optimization=self.enable_all_optimization,
            enable_fused_normalization=self.enable_fused_normalization,
            enable_flash_attention=self.enable_flash_attention,
            enable_jit_fused=self.enable_jit_fused,
            enable_sequence_parallelism=enable_sequence_parallelism,
            
            # sequence_parallelism_mode=sequence_parallelism_mode,
            # enable_sequence_overlap=enable_sequence_overlap,
            # parallel_output=parallel_output,
            # make_vocab_size_divisible_by=make_vocab_size_divisible_by,
            # gradient_checkpoint_config=gradient_checkpoint_config,
            extra_kwargs=sharding_extra_kwargs,
        )

        if self.custom_policy is not None:
            self.custom_policy.set_shard_config(self.shard_config)
        # self.amp_config = dict(
        #     initial_scale=initial_scale,
        #     growth_factor=growth_factor,
        #     backoff_factor=backoff_factor,
        #     growth_interval=growth_interval,
        #     hysteresis=hysteresis,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        # )

        # self.ddp_config = dict(
        #     broadcast_buffers=broadcast_buffers,
        #     bucket_cap_mb=ddp_bucket_cap_mb,
        #     find_unused_parameters=find_unused_parameters,
        #     check_reduction=check_reduction,
        #     gradient_as_bucket_view=gradient_as_bucket_view,
        #     static_graph=static_graph,
        # )

        # self.zero_config = dict(
        #     reduce_bucket_size=zero_bucket_size_in_m * 1024 * 1024,
        #     communication_dtype=communication_dtype,
        #     overlap_communication=overlap_communication,
        #     cpu_offload=cpu_offload,
        #     partition_grad=(self.zero_stage == 2),
        #     forced_dtype=PRECISION_TORCH_TYPE[precision],
        # )

        self.max_norm = max_norm

    def __del__(self):
        """Destroy the process groups in ProcessGroupMesh"""
        self.pg_mesh.destroy_mesh_process_groups()

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return self.pp_size > 1

    def supported_devices(self) -> List[str]:
        return ["cuda", "npu"]

    def supported_precisions(self) -> List[str]:
        return ["fp16", "bf16", "fp32"]

    def control_device(self) -> bool:
        return True

    def control_precision(self) -> bool:
        return True

    def support_no_sync(self) -> bool:
        return True

    def support_lora(self) -> bool:
        return False

    def control_checkpoint_io(self) -> bool:
        return True

    def configure(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None
    ) -> Tuple[Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        # param_info = get_param_info(optimizer)

        # TODO: Support Galore + ZeRO
        # zero_stage = self.zero_stage
        # zero_config = deepcopy(self.zero_config)

        # Replace with distributed implementation if exists
        # optimizer = cast_to_distributed(optimizer)

        # if isinstance(optimizer, DistGaloreAwamW) and zero_stage > 0 and self.dp_size > 0:
        #     warnings.warn("Galore is only supported for Tensor Parallel and vanilla Data Parallel yet. Disabling ZeRO.")
        #     # zero_config["partition_grad"] = False
        #     zero_stage = 0

        if not isinstance(model, ModelWrapper):
            # use_ddp = (self.dp_size > 1 and self.pp_size == 1 and self.zero_stage == 0) or (
            #     self.dp_size == 1
            #     and self.pp_size == 1
            #     and self.enable_sequence_parallelism
            #     and self.sequence_parallelism_mode == "all_to_all"
            # )
            # if self.enable_sequence_parallelism and self.sequence_parallelism_mode == "all_to_all":
            #     dp_group = self.pg_mesh.create_group_along_axis([self.dp_axis, self.sp_axis])
            # else:
            dp_group = self.dp_group
            model = HybridParallelInferenceModule(
                model,
                precision=self.precision,
                shard_config=self.shard_config,
                dp_group=dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
                # use_ddp=use_ddp,
                # ddp_config=self.ddp_config,
                custom_policy=self.custom_policy,
            )
       
        return model, optimizer, criterion, dataloader, lr_scheduler

    def execute_pipeline(
        self,
        input_object: dict,
        model: HybridParallelInferenceModule,
        criterion: Callable[[Any, Any], torch.Tensor] = None,
        # optimizer: Optional[
        #     Union[HybridParallelNaiveOptimizer, HybridParallelAMPOptimizer, HybridParallelZeroOptimizer]
        # ] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        # assert self.enable_pipeline_parallelism, "pipeline parallelism is not enabled"

        if return_outputs:
            warnings.warn("return_outputs may lead to significant extra memory consumption.")

        # Create a context for gradient synchronization based on the optimizer type.
        # If it's a HybridParallelZeroOptimizer, use optimizer.no_sync(); otherwise, use model.no_sync().
        # This is to avoid redundant gradient reduction in pipeline parallelism (multiple microbatch values should be reduced once),
        # so we disable it, performing manual reduction instead.
        # ctx = optimizer.no_sync() if isinstance(optimizer, HybridParallelZeroOptimizer) else model.no_sync()

        # with ctx:
        with torch.no_grad():
            # print("======= input_object =========== ", input_object['input_tokens_ids'].shape)
            outputs = self.schedule.forward_backward_step(
                model, input_object, criterion, None, return_loss, return_outputs
            )
            # print("======= outputs =========== ", outputs)
            

        # run with gradients accumulation
        # if model.require_grad_sync == False or (
        #     isinstance(optimizer, HybridParallelZeroOptimizer) and optimizer._grad_store.require_grad_sync == False
        # ):
            return outputs

        # # Synchronize the grads of shared parameters of the model.
        # model.sync_shared_params()
        # # Synchronize sequence parallelism gradients of the model.
        # model.sync_sp_grads()

        # # Check if the optimizer is a HybridParallelZeroOptimizer and synchronize data parallelism gradients if so.
        # # Otherwise, synchronize data parallelism gradients of the model.
        # # This is because these are two different forms of data parallelism.
        # if isinstance(optimizer, HybridParallelZeroOptimizer):
        #     optimizer.sync_dp_grads()
        # else:
        #     model.sync_dp_grads()

        # return outputs

    def prepare_dataloader(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=1024,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        distributed_sampler_cls=None,
        **kwargs,
    ):
        r"""
        Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader` and `torch.utils.data.DistributedSampler`.


        Args:
            dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random worker seed for sampling, defaults to 1024.
            add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
            drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
                is not divisible by the batch size. If False and the size of dataset is not divisible by
                the batch size, then the last batch will be smaller, defaults to False.
            pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
            num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
            kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                    `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

        Returns:
            :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
        """
        _kwargs = kwargs.copy()
        distributed_sampler_cls = distributed_sampler_cls or DistributedSampler
        sampler = distributed_sampler_cls(
            dataset,
            num_replicas=self.pg_mesh.size(self.dp_axis),
            rank=self.pg_mesh.coordinate(self.dp_axis),
            shuffle=shuffle,
        )

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )

    def get_checkpoint_io(self) -> CheckpointIO:
        return HybridParallelCheckpointIO(self.dp_group, self.pp_group, self.tp_group, self.zero_stage)

    def no_sync(self, model: Module, optimizer: OptimizerWrapper) -> Iterator[None]:
        assert (
            self.zero_stage != 2
        ), "ZERO2 is not compatible with no_sync function, please run gradient accumulation with gradient synchronization allowed."
        return optimizer.no_sync() if isinstance(optimizer, HybridParallelZeroOptimizer) else model.no_sync()

    def enable_lora(
        self, model: Module, pretrained_dir: Optional[str] = None, lora_config: Optional[Dict] = None
    ) -> Module:
        raise NotImplementedError
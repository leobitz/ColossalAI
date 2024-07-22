from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import tree_map
from torch import distributed as dist
from colossalai.accelerator import get_accelerator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication, create_send_metadata
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.utils import get_current_device
from torch.utils._pytree import SUPPORTED_NODES, TreeSpec, _register_pytree_node, tree_flatten, tree_map, tree_unflatten


from colossalai.pipeline.schedule._utils import (
    detach,
    get_batch_size,
    get_micro_batch,
    merge_batch,
    model_forward,
    retain_grad,
    to_device,
    tree_map_hf,
)
from colossalai.pipeline.schedule.base import PipelineSchedule


class OneForwardOneBackwardSchedule(PipelineSchedule):
    def __init__(
        self,
        stage_manager: PipelineStageManager,
        num_microbatches: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        enable_metadata_cache: bool = True,
    ) -> None:
        """1F1B pipeline schedule.

        Args:
            stage_manager (PipelineStageManager): Pipeline stage manager
            num_microbatches (Optional[int], optional): The number of microbatches. If not provided, it will be derived from microbatch size. Defaults to None.
            microbatch_size (Optional[int], optional): Microbatch size. If num_microbatches is provided, this will be ignored. Defaults to None.
        """
        super().__init__(stage_manager)
        assert (
            num_microbatches is not None or microbatch_size is not None
        ), "Either num_microbatches or microbatch_size should be provided"

        self.comm = PipelineP2PCommunication(stage_manager)
        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

        # P2PMeta cache
        self.enable_metadata_cache = enable_metadata_cache
        self.send_tensor_metadata = True
        self.send_grad_metadata = True
        self.tensor_metadata_recv = None
        self.grad_metadata_recv = None

    def load_batch(self, input_obj: dict, device: Optional[torch.device] = None) -> None:
        """Load a batch from data iterator.

        Args:
            data_iter (Iterable): Data iterator.
            device (Optional[torch.device], optional): Target device. Defaults to None.
        """
        batch = input_obj#next(input_obj)
        if device is not None:
            batch = tree_map(partial(to_device, device=device), batch)


        self.microbatch_offset = 0
        self.batch = batch
        self.batch_size = get_batch_size(batch['input_tokens_ids'])
        # data_list, _ = tree_flatten(batch)
        # print([data.shape if isinstance(data, torch.Tensor) else type(data) for data in data_list])
        # print("Batch size", self.batch_size)
        if self.microbatch_size is None:
            assert self.batch_size % self.num_microbatches == 0, "Batch size should divided by # microbatches"
            self.microbatch_size = self.batch_size // self.num_microbatches
        if self.num_microbatches is None:
            assert self.batch_size % self.microbatch_size == 0, "Batch size should divided by the microbatch size"
            self.num_microbatches = self.batch_size // self.microbatch_size

        if not self.forward_only:
            assert self.last_batch_size is None or self.last_batch_size == self.batch_size
            assert self.batch_size == self.microbatch_size * self.num_microbatches

            assert (
                self.num_microbatches >= self.stage_manager.num_stages
            ), "Number of microbatch should be larger than number of stages"

        if self.forward_only:
            self.num_microbatches = (self.batch_size - 1) // self.microbatch_size + 1
            # NOTE: disable metadata cache when batch size changes (not valid anymore)
            if self.batch_size != self.last_batch_size:
                self.enable_metadata_cache = False
                self.send_tensor_metadata = True
                self.send_grad_metadata = True
                self.tensor_metadata_recv = None
                self.grad_metadata_recv = None

        self.last_batch_size = self.batch_size

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.

        Returns:
            Any: Micro batch.
        """
        assert self.microbatch_offset <= self.batch_size, "Microbatches exhausted"
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset, self.microbatch_size)
        self.microbatch_offset += self.microbatch_size
        return tree_map(partial(to_device, device=get_accelerator().get_current_device()), micro_batch)

    def recv_forward(self, prev_rank: int = None) -> Any:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.
           For 1F1B.

        Args:
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
        """
        if not self.stage_manager.is_first_stage():
            input_tensor = self.comm.recv_forward(prev_rank, metadata_recv=self.tensor_metadata_recv)
            if self.enable_metadata_cache and self.tensor_metadata_recv is None:
                self.tensor_metadata_recv = create_send_metadata(input_tensor)

            return input_tensor

    # def recv_backward(self, next_rank: int = None) -> Any:
    #     """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
    #        For 1F1B.

    #     Args:
    #         next_rank (int, optional): The rank of the source of the tensor.

    #     Returns:
    #         Any: The input gradient tensor or gradient tensor list.
    #     """
    #     if not self.stage_manager.is_last_stage():
    #         output_tensor_grad = self.comm.recv_backward(next_rank, metadata_recv=self.grad_metadata_recv)
    #         if self.enable_metadata_cache and self.grad_metadata_recv is None:
    #             self.grad_metadata_recv = create_send_metadata(output_tensor_grad)

    #         return output_tensor_grad

    def send_forward(self, output_tensor: Any, next_rank: int = None) -> None:
        """Sends the input tensor to the next stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.stage_manager.is_last_stage():
            self.comm.send_forward(output_tensor, next_rank, send_metadata=self.send_tensor_metadata)
            self.send_tensor_metadata = not self.enable_metadata_cache

    # def send_backward(self, input_tensor_grad: Any, prev_rank: int = None) -> None:
    #     """Sends the gradient tensor to the previous stage in pipeline.
    #        For 1F1B.

    #     Args:
    #         input_object (Any): Object to be sent.
    #         prev_rank (int, optional): The rank of the recipient of the tensor
    #     """
    #     if not self.stage_manager.is_first_stage():
    #         self.comm.send_backward(input_tensor_grad, prev_rank, send_metadata=self.send_grad_metadata)
    #         self.send_grad_metadata = not self.enable_metadata_cache

    # def send_forward_recv_backward(
    #     self, output_tensor: Any, next_rank: int = None, send_prior_fallback: Optional[bool] = None
    # ) -> Any:
    #     """Sends the input tensor to the next stage and copy the gradient tensor from the next stage in pipeline.
    #        For 1F1B.

    #     Args:
    #         output_object (Any): Object to be sent.
    #         next_rank (int, optional): The rank of the recipient of the tensor.
    #     """
    #     if not self.stage_manager.is_last_stage():
    #         if not self.send_tensor_metadata and self.grad_metadata_recv is not None:
    #             send_prior_fallback = None  # must not fallback
    #         output_tensor_grad = self.comm.send_forward_recv_backward(
    #             output_tensor,
    #             next_rank,
    #             send_metadata=self.send_tensor_metadata,
    #             metadata_recv=self.grad_metadata_recv,
    #             send_prior_fallback=send_prior_fallback,
    #         )
    #         self.send_tensor_metadata = not self.enable_metadata_cache
    #         if self.enable_metadata_cache and self.grad_metadata_recv is None:
    #             self.grad_metadata_recv = create_send_metadata(output_tensor_grad)

    #         return output_tensor_grad

    # def send_backward_recv_forward(
    #     self, input_tensor_grad: Any, prev_rank: int = None, send_prior_fallback: Optional[bool] = None
    # ) -> Any:
    #     """Sends the gradient tensor to the previous stage and copy the input tensor from the previous stage in pipeline.
    #        For 1F1B.

    #     Args:
    #         output_object (Any): Object to be sent.
    #         prev_rank (int, optional): The rank of the recipient of the tensor.
    #     """
    #     if not self.stage_manager.is_first_stage():
    #         if not self.send_grad_metadata and self.tensor_metadata_recv is not None:
    #             send_prior_fallback = None  # must not fallback
    #         input_tensor = self.comm.send_backward_recv_forward(
    #             input_tensor_grad,
    #             prev_rank,
    #             send_metadata=self.send_grad_metadata,
    #             metadata_recv=self.tensor_metadata_recv,
    #             send_prior_fallback=send_prior_fallback,
    #         )
    #         self.send_grad_metadata = not self.enable_metadata_cache
    #         if self.enable_metadata_cache and self.tensor_metadata_recv is None:
    #             self.tensor_metadata_recv = create_send_metadata(input_tensor)

    #         return input_tensor

    def forward_step(
        self,
        model: Module,
        input_obj: Optional[dict],
        other_obj: Optional[dict],
        criterion: Callable=None,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[List[Any]] = None,
    ) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline

        Args:
            model (Module): Model to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        micro_batch = self.load_micro_batch()
        # print(micro_batch['input_tokens_ids'].shape, self.microbatch_size, self.num_microbatches, self.batch_size)
        # for the first stage, input_obj is None
        # for the non-first stage, input_obj is the output of the previous stage and it's must be a dict
        # out_dict = {}
        # for k, v in micro_batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"micro_batch[{k}]: {v.size()}")
        # print("Before model_forward", self.stage_manager.stage)
        # print("Microbaching")
        # print("Micro", [type(item) for item in micro_batch])
        # if input_obj is not None:
        #     for item in input_obj:
        #         if type(item) == dict:
        #             print(" -> dict -> Input", [(key, type(item[key])) for key in item.keys()])
        #         elif type(item) == list:
        #             print(" -> list -> Input", [type(i) for i in item])
        #         else:
        #             print(" -> don't know -> Input", type(item))
        # else:
        #     print("Input", input_obj)
        # print("Before slicing", micro_batch['input_tokens_ids'].shape)
        for key in micro_batch:
            if key == 'input_tokens_ids':
                micro_batch['input_tokens_ids'] = micro_batch['input_tokens_ids'].flatten()
            elif key == 'output_tensor':
                micro_batch['output_tensor'] = micro_batch['output_tensor'].reshape(-1, micro_batch['output_tensor'].shape[-1])
        # print(micro_batch.keys())
        # print(micro_batch['input_tokens_ids'], "micro_batch", self.stage_manager.stage)
        # micro_batch = tree_map(_get_tensor_slice, micro_batch)
        # print("After slicing", micro_batch['input_tokens_ids'].shape, self.stage_manager.stage)
        # print(input_obj, "Input objjj Input objjj")
        # merge micro_batch with other_obj dicts
        # print("other dict keys", other_obj.keys())
        # print("micro_batch keys", micro_batch.keys())
        # if input_obj is not None:
        #     print("input_obj keys", input_obj.keys())
        micro_batch.update(other_obj)
        # print(micro_batch['inputmetadata'].block_tables)
        output_obj = model_forward(model, micro_batch, input_obj)
        # print("output_obj ",type( output_obj), self.stage_manager.stage)
        # print("After model_forward", self.stage_manager.stage, type(output_obj))
        # if self.stage_manager.is_last_stage():
        #     if outputs is not None:
        #         outputs.append(tree_map_hf(detach, output_obj))
        #     # if type(output_obj) == dict:
        #     #     print("output_obj[key].shape", [output_obj[key].shape for key in output_obj.keys()])
        #     # else:
        #     #     print("output_obj.shape", output_obj.shape)
        #     return outputs
        # else:
            # print("output_obj[key].shape", [output_obj[key].shape for key in output_obj.keys()])
        return output_obj

    # def backward_step(
    #     self,
    #     optimizer: OptimizerWrapper,
    #     input_obj: Optional[dict],
    #     output_obj: Union[dict, torch.Tensor],
    #     output_obj_grad: Optional[dict],
    # ) -> Optional[dict]:
    #     """Backward one step of the pipeline

    #     Args:
    #         optimizer (OptimizerWrapper): Optimizer to update the model
    #         input_obj (Optional[dict]): Output of the previous stage. If it is the first stage, the `input_obj` is None.
    #         output_obj (Union[dict, torch.Tensor]): Output of the current stage. If it is the last stage, the output is the loss (Tensor).
    #         output_obj_grad (dict): Gradient of the `output_obj`. If it is the last stage, the `output_obj_grad` is None.

    #     Returns:
    #         Optional[dict]: Gradient of the `input_obj`. If it is the first stage, the `input_obj_grad` is None.
    #     """

    #     # Retain the grad on the input_obj.
    #     tree_map(retain_grad, input_obj)
    #     # Backward pass.
    #     if output_obj_grad is None:
    #         optimizer.backward(output_obj)
    #     else:
    #         if "backward_tensor_keys" not in output_obj:
    #             for k, grad in output_obj_grad.items():
    #                 optimizer.backward_by_grad(output_obj[k], grad)
    #         else:
    #             for k, grad in output_obj_grad.items():
    #                 output_obj[k].grad = grad
    #             for k in output_obj["backward_tensor_keys"]:
    #                 tensor_to_backward = output_obj[k]
    #                 optimizer.backward_by_grad(tensor_to_backward, tensor_to_backward.grad)

    #     # Collect the grad of the input_obj.
    #     input_obj_grad = None
    #     if input_obj is not None:
    #         input_obj_grad = {}
    #         for k, v in input_obj.items():
    #             if isinstance(v, torch.Tensor) and v.grad is not None:
    #                 input_obj_grad[k] = v.grad
    #     return input_obj_grad

    def run_forward_only(
        self,
        model: Module,
        input_obj: dict,
        criterion: Callable[..., Any]=None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        """
        Runs forward only schedule, with communication between pipeline stages.
        """
        assert self.forward_only

        input_dict = {"input_tokens_ids": input_obj["input_tokens_ids"], 'output_tensor': input_obj['output_tensor']}
        other_dict = {key: value for key, value in input_obj.items() if key not in ['input_tokens_ids', 'output_tensor']}
        # print(other_dict.keys())
        self.load_batch(input_dict)

        accum_loss = None
        # if return_loss and self.stage_manager.is_last_stage():
            # accum_loss = torch.scalar_tensor(0, device=get_accelerator().get_current_device())
        # outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None
        # print("======= num_microbatches =========== ", self.num_microbatches, [key for key in input_obj.keys()])
        for _ in range(self.num_microbatches):
            input_obj = self.recv_forward()
            # print("input_obj", "input_obj", self.stage_manager.stage)
            output_obj = self.forward_step(model, input_obj,  other_dict, None, None, None)
            # print(f"{_} x-> output_objxx", self.stage_manager.stage, type(output_obj))
            self.send_forward(output_obj)
            # print(f"{_} x-> output_obj", output_obj.shape)
        # return output_obj
            # if type(output_obj) == dict:
            #     print("======= outputs =========== dict", self.stage_manager.stage, type(output_obj))
            # elif type(output_obj) == torch.Tensor:
            #     print("======= outputs =========== tensor", self.stage_manager.stage, output_obj.shape)
            # else:
            #     print("======= outputs =========== else", self.stage_manager.stage, type(output_obj))
        
        # get number of gpus
        # num_gpus = dist.get_world_size()

        # if num_gpus > 1:
        if self.stage_manager.is_last_stage():
            rank = dist.get_rank()
            # to_rank = 0 if rank % 2 == 0 else 1
            # if rank == 2:
            #     to_rank = 0
            #     self.comm.send_backward(output_obj, to_rank)
            # if rank == 3:
            #     self.comm.send_backward(output_obj, 1)
            # self.comm.send_backward(output_obj, 2)
            self.comm.send_backward(output_obj, 0)
            self.comm.send_backward(output_obj, 1)
            self.comm.send_backward(output_obj, 2)
            # print("just sent it", self.stage_manager.stage)
            return output_obj

            # print("# if rank is 0, then receive the last stage output")
        if not self.stage_manager.is_last_stage():
            # print("======= recv_forwardx =========== ", self.stage_manager.stage)
            # from_rank = 2 if dist.get_rank() == 0 else 3
            # from_rank = dist.get_world_size() - 1
            # if dist.get_rank() == 0:
            #     outputs = self.comm.recv_backward(2)
            # if dist.get_rank() == 1:
            #     outputs = self.comm.recv_backward(3)
            outputs = self.comm.recv_backward(3)
            # print("just received it", self.stage_manager.stage)
            # print(type(output_obj),output_obj.keys())
            return outputs[0]
            # print("======= recv_forwardy =========== ", self.stage_manager.stage, outputs[0].shape)
        # logit_from_last_stage = self.recv_forward()

        # if self.stage_manager.is_last_stage():
        #     return None

        # if outputs is not None:
        #     if isinstance(model, ModelWrapper):
        #         model = model.unwrap()
        #     batch_size_dim = getattr(model, "batch_size_dim", 0)
        #     outputs = merge_batch(outputs, batch_size_dim)
        # return {"outputs": outputs}

    # def run_forward_backward(
    #     self,
    #     model: Module,
    #     data_iter: Iterable,
    #     criterion: Callable[..., Any],
    #     optimizer: Optional[OptimizerWrapper] = None,
    #     return_loss: bool = False,
    #     return_outputs: bool = False,
    # ) -> Dict:
    #     """
    #     Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
    #     """
    #     assert not self.forward_only

    #     self.load_batch(data_iter)

    #     # num_warmup_microbatches is the step when not all the processes are working
    #     num_warmup_microbatches = self.stage_manager.num_stages - self.stage_manager.stage - 1
    #     num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
    #     num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

    #     # Input, output tensors only need to be saved when doing backward passes
    #     input_objs, output_objs = [], []

    #     accum_loss = None
    #     if return_loss and self.stage_manager.is_last_stage():
    #         accum_loss = torch.scalar_tensor(0, device=get_current_device())
    #     outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

    #     # Run warmup forward passes.
    #     for i in range(num_warmup_microbatches):
    #         input_obj = self.recv_forward()
    #         output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
    #         self.send_forward(output_obj)
    #         input_objs.append(input_obj)
    #         output_objs.append(output_obj)

    #     # Before running 1F1B, need to receive first forward tensor.
    #     # If all microbatches are run in warmup / cooldown phase, then no need to
    #     # receive this tensor here.
    #     if num_microbatches_remaining > 0:
    #         input_obj = self.recv_forward()

    #     # Run 1F1B in steady state.
    #     for i in range(num_microbatches_remaining):
    #         last_iteration = i == (num_microbatches_remaining - 1)

    #         output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
    #         output_obj_grad = self.send_forward_recv_backward(
    #             output_obj, send_prior_fallback=self.stage_manager.stage % 2 == 0
    #         )
    #         # Add input_obj and output_obj to end of list.
    #         input_objs.append(input_obj)
    #         output_objs.append(output_obj)

    #         # Pop output_obj and output_obj from the start of the list for
    #         # the backward pass.
    #         input_obj = input_objs.pop(0)
    #         output_obj = output_objs.pop(0)
    #         input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)

    #         if last_iteration:
    #             self.send_backward(input_obj_grad)
    #         else:
    #             input_obj = self.send_backward_recv_forward(
    #                 input_obj_grad, send_prior_fallback=self.stage_manager.stage % 2 == 0
    #             )

    #     # Run cooldown backward passes.
    #     for i in range(num_warmup_microbatches):
    #         input_obj = input_objs.pop(0)
    #         output_obj = output_objs.pop(0)

    #         output_obj_grad = self.recv_backward()
    #         input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)
    #         self.send_backward(input_obj_grad)

    #     assert all(len(v) == 0 for v in input_objs) and all(len(v) == 0 for v in output_objs)

    #     if outputs is not None:
    #         if isinstance(model, ModelWrapper):
    #             model = model.unwrap()
    #         batch_size_dim = getattr(model, "batch_size_dim", 0)
    #         outputs = merge_batch(outputs, batch_size_dim)
    #     return {"loss": accum_loss, "outputs": outputs}

    def forward_backward_step(
        self,
        model: Module,
        input_obj: dict,
        criterion: Callable[..., Any]=None,
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        """
        Args:
            model (Module): Model to be trained.
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            optimizer (OptimizerWrapper, optional): Optimizer to be used. Can be None when only forward is executed. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: Dictionary containing loss and outputs.
        """
        
        self.forward_only = not torch.is_grad_enabled()
        if optimizer is None:
            assert self.forward_only, "Optimizer should be passed when doing backward."
        # print("Forward only ", "stage", self.stage_manager.stage)
        # if self.stage_manager.is_last_stage():
        #     rank = dist.get_rank()
            # print("First stage with rank ", rank)
        result = self.run_forward_only(model, input_obj, criterion, return_loss, return_outputs)
        
        return result

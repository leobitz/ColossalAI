import argparse

import colossalai.inference
import colossalai.inference.utils
from torch import bfloat16, float16, float32
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import colossalai
from contextlib import nullcontext
from colossalai.lazy import LazyInitContext
from colossalai.accelerator import get_accelerator
from colossalai.cluster import DistCoordinator
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.modeling.policy.nopadding_llama import NoPaddingLlamaModelInferPolicy
from colossalai.legacy.pipeline.pipelinable import PipelinableContext
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.context import ParallelMode
from colossalai.inference.core import HybridParallelInferencePlugin
# For Llama 3, we'll use the following configuration
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM
# from performance_evaluator import PerformanceEvaluator, get_profile_context
import resource
# from model_utils import format_numel_str, get_model_numel
from colossalai.booster import Booster
from colossalai.inference.utils import get_model_size, has_index_file
import sys
import pandas as pd
import time
import torch.distributed as dist

MODEL_CLS = AutoModelForCausalLM
POLICY_CLS = NoPaddingLlamaModelInferPolicy

TORCH_DTYPE_MAP = {
    "fp16": float16,
    "fp32": float32,
    "bf16": bfloat16,
}

# MODEL_CONFIGS = {
#     "7b": LlamaConfig(max_position_embeddings=4096),
#     "13b": LlamaConfig(
#         hidden_size=5120,
#         intermediate_size=13824,
#         num_hidden_layers=40,
#         num_attention_heads=40,
#         max_position_embeddings=4096,
#     ),
#     "70b": LlamaConfig(
#         hidden_size=8192,
#         intermediate_size=28672,
#         num_hidden_layers=80,
#         num_attention_heads=64,
#         max_position_embeddings=4096,
#         num_key_value_heads=8,
#     ),
# }

def infer(args):
    # args.config = "7b"
    args.dtype = "bf16"
    # ==============================
    # Launch colossalai, setup distributed environment
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()
    
    # if args.config in MODEL_CONFIGS:
    #     config = MODEL_CONFIGS[args.config]
    # else:
    
    if args.pp_size == 1:
        args.mbs = args.max_batch_size
    else:
        args.mbs = args.max_batch_size // args.pp_size
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # ckpt config for LLaMA3-70B on 64 H100 GPUs
    hybrid_kwargs = (
        # {
        #     # "gradient_checkpoint_config": PipelineGradientCheckpointConfig(
        #     #     num_ckpt_layers_per_stage=[19, 19, 19, 13],
        #     # ),
        #     "num_layers_per_stage": [19, 20, 20, 21],
        # }
        # if args.custom_ckpt
        # else {}
    )
    args.use_cuda_kernel = True
    inference_config = InferenceConfig(
        dtype=TORCH_DTYPE_MAP.get(args.dtype, None),
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        prefill_ratio=1.2,
        block_size=16,
        tp_size=args.tp_size,
        use_cuda_kernel=args.use_cuda_kernel,
        enable_streamingllm=args.enable_streamingllm,
        start_token_size=args.start_token_size,
        generated_token_size=args.generated_token_size,
    )
    # print("Microbatches ", args.mbs)
    model_shard_infer_config = inference_config.to_model_shard_inference_config()
    # model_shard_infer_config.use_cuda_kernel = False
    
    policy = POLICY_CLS()
    plugin = HybridParallelInferencePlugin(args.tp_size, args.pp_size,
            # zero_stage=args.zero,
            # sp_size=args.sp,
            enable_all_optimization=False,
            enable_sequence_parallelism=False,
            enable_fused_normalization=False,#torch.cuda.is_available(),
            enable_flash_attention=False,
            enable_jit_fused=False,
            microbatch_size=args.mbs,
            num_microbatches=args.max_batch_size // args.mbs,
            precision=args.dtype,
            dp_outside=False,
            sharding_extra_kwargs={"model_shard_infer_config": model_shard_infer_config},
            custom_policy=policy)
    init_ctx = (
        LazyInitContext(default_device=get_accelerator().get_current_device())
        if isinstance(plugin, (HybridParallelInferencePlugin,))
        else nullcontext()
    )
    
    init_kwargs = {}
    if config.model_type == "chatglm":
        init_kwargs["empty_init"] = False

    with init_ctx:
        # model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, **init_kwargs)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=TORCH_DTYPE_MAP.get(args.dtype, None))
        model = model.to(TORCH_DTYPE_MAP.get(args.dtype, None)).eval()
        # memory_usage_bytes = sum(sys.getsizeof(param.storage()) for param in model.parameters())
        # memory_usage_gb = memory_usage_bytes / (1024 ** 3)
        # print(memory_usage_gb, "GB")
    # model_numel = get_model_numel(model)
    # coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")
    # performance_evaluator = PerformanceEvaluator(
    #     model_numel,
    #     model.config.num_hidden_layers,
    #     model.config.hidden_size,
    #     model.config.vocab_size,
    #     args.grad_checkpoint,
    #     args.ignore_steps,
    #     dp_world_size=dp_size,
    # )
    booster = Booster(plugin=plugin)
    # optimizer = HybridAdam(model.parameters())
    # torch.set_default_dtype(torch.bfloat16)
    model, _, _, _, _ = booster.boost(model)
    # memory_usage_bytes = sum(sys.getsizeof(param.storage()) for param in model.parameters())
    # memory_usage_bytes = torch.cu`da.memory_allocated()
    # memory_usage_gb = memory_usage_byt`es / (1024 ** 3)
    # print(memory_usage_gb, "GB")
    # torch.set_default_dtype(torch.float)
    coordinator.print_on_master(
        f"Booster init max CUDA memory: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB"
    )
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )
    
    model_path_or_name = args.model
    # pipelinable = PipelinableContext()
    # with pipelinable:
    #     model = MODEL_CLS.from_pretrained(model_path_or_name, torch_dtype=TORCH_DTYPE_MAP.get(args.dtype, None))
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    tokenizer.pad_token = tokenizer.eos_token
    # coordinator.print_on_master(f"Model Config:\n{model.config}")

    # pipelinable.to_layer_list()
    # pipelinable.policy = "uniform"
    # model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

    # count number of parameters
    # total_numel = 0
    # for p in model.parameters():
    #     total_numel += p.numel()
    # if not gpc.is_initialized(ParallelMode.PIPELINE):
    #     pipeline_stage = 0
    # else:
    #     pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    # print(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")


    # ==============================
    # Initialize InferenceEngine
    # ==============================
    

    # hybrid_plugin  = HybridParallelInferencePlugin(args.tp_size, args.pp_size)
    coordinator.print_on_master(f"Initializing Inference Engine...")
    engine = InferenceEngine(model, tokenizer, inference_config, config, model_policy=POLICY_CLS(),
                              verbose=True,
                              hybrid_inference=plugin,
                              hybrid_model=model,
                              tp_size=args.tp_size,
                              pp_size=args.pp_size,
                            )

    # ==============================
    # Generation
    # ==============================
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )
    # prompts = pd.read_csv("prompts.csv")['prompt'].values.flatten().tolist()
    # prompts = [prompt.split(" ") for prompt in prompts]
    # # print(prompts[:4])
    # # print()
    # # prompts = list(filter(lambda x: len(x) < 10 and len(x) > 0, prompts))
    # prompts = [" ".join(prompt[:10]) for prompt in prompts]
    # # print(len(prompts))
    # prompts = prompts[:128]
    prompts = [args.prompt] * 256
    
    coordinator.print_on_master(f"Generating...")
    # print time on master for benchmarking
    start = time.time()
    out = engine.generate(prompts=prompts, generation_config=generation_config)
    end = time.time()
    if coordinator.is_master():
        elapsed = end - start
        line = f"{args.tp_size},{args.pp_size},{args.mbs},{args.max_batch_size},{elapsed}\n"
        open("results.csv", "a").write(line)
    coordinator.print_on_master(out)
    # ==============================
    # Optionally, load drafter model and proceed speculative decoding
    # ==============================
    # drafter_model_path_or_name = args.drafter_model
    # if drafter_model_path_or_name is not None:
    #     drafter_model = AutoModelForCausalLM.from_pretrained(drafter_model_path_or_name)
    #     # turn on speculative decoding with the drafter model
    #     engine.enable_spec_dec(drafter_model)
    #     coordinator.print_on_master(f"Generating...")
    #     out = engine.generate(prompts=[args.prompt], generation_config=generation_config)
    #     coordinator.print_on_master(out)

    #     engine.disable_spec_dec()


# colossalai run --nproc_per_node 1 llama_generation.py -m MODEL_PATH
# colossalai run --nproc_per_node 2 llama_generation.py -m MODEL_PATH --tp_size 2
if __name__ == "__main__":
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to the model or model name")
    parser.add_argument("--drafter_model", type=str, help="Path to the drafter model or model name")
    parser.add_argument(
        "-p", "--prompt", type=str, default="Introduce some landmarks in the United Kingdom, such as", help="Prompt"
    )
    parser.add_argument("-b", "--max_batch_size", type=int, default=2, help="Max batch size")
    parser.add_argument("-i", "--max_input_len", type=int, default=128, help="Max input length")
    parser.add_argument("-o", "--max_output_len", type=int, default=128, help="Max output length")
    parser.add_argument("-t", "--tp_size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("--mbs", type=int, default=1, help="micro-batche size")
    parser.add_argument("-q", "--pp_size", type=int, default=1, help="Pipeline Parallelism size")
    parser.add_argument("-d", "--dtype", type=str, default="fp16", help="Data type", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--use_cuda_kernel", action="store_true", help="Use CUDA kernel, use Triton by default")
    # Generation configs
    parser.add_argument("--max_length", type=int, default=64, help="Max length for generation")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=50, help="Top k for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p for generation")
    parser.add_argument("--enable_streamingllm", action="store_true", help="Whether to use StreamingLLM")
    parser.add_argument(
        "--start_token_size", type=int, default=4, help="The size of the start_token, When using StreamingLLM,"
    )
    parser.add_argument(
        "--generated_token_size", type=int, default=512, help="The size of the generated_token, When using StreamingLLM"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="If no_repeat_ngram_size > 0, the consecutive tokens of ngram size can only appear once in inference sentences.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter that influences the model's treatment of new tokens in relation to their appearance in the prompt and the generated text. Values greater than 1 incentivize the model to introduce new tokens, whereas values less than 1 incentivize token repetition., defaults to 1.0.",
    )
    args = parser.parse_args()
    with torch.no_grad():
        infer(args)

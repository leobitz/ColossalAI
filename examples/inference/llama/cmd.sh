
# create a forloop [1, 2, 4, 8, 16, 32, 64]
for i in 128 256
do
    # colossalai run --nproc_per_node 4 llama_generation_pipeline2.py -m meta-llama/Llama-2-13b-hf --max_length 32 --tp_size 4 --pp_size 1 -b $i

    # colossalai run --nproc_per_node 2 llama_generation_pipeline2.py -m meta-llama/Llama-2-13b-hf --max_length 32 --tp_size 2 --pp_size 1 -b $i

    # colossalai run --nproc_per_node 1 llama_generation_pipeline2.py -m meta-llama/Meta-Llama-3-8B --max_length 32 --tp_size 1 --pp_size 1 -b $i 

    # colossalai run --nproc_per_node 4 llama_generation_pipeline2.py -m meta-llama/Llama-2-13b-hf --max_length 32 --tp_size 2 --pp_size 2 -b $i

    colossalai run --nproc_per_node 4 llama_generation_pipeline2.py -m meta-llama/Llama-2-13b-hf --max_length 32 --tp_size 1 --pp_size 4 -b $i
done

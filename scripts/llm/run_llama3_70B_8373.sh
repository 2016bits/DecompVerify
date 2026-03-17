export CUDA_VISIBLE_DEVICES=3
python -O -u -m vllm.entrypoints.openai.api_server \
        --host=127.0.0.1 \
        --port=8373 \
        --model=Meta-Llama-3-70B-Instruct-AutoAWQ-4bit \
        --tokenizer=meta-llama/Meta-Llama-3-70B-Instruct \
        --tensor-parallel-size=1 \
        --quantization awq \
        --dtype half    \
        --gpu-memory-utilization 0.9

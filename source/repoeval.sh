export LC_ALL="POSIX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python repoeval.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-0.5B \
    --input_file data/repoeval/python_TASK_sparse_oracle.jsonl \
    --model_output_dir model_outputs/qwen25-05-base \
    --eval_output_dir eval_outputs/qwen25-05-base \
    --tasks line_completion function_completion api_completion \
    --ts_lib grammar/LANGUAGE-lang-parser.so \
    --max_seq_length 8096 \
    --gen_length 50 \
    --cfc_seq_length 2048 \
    --right_context_length 2048 \
    --tp 1 \
    --vram_utilization 0.9 \
    # --lora_path /home/jovyan/anh-uet/models/loras/qwen25-15i-psm-line-tpu
    
    
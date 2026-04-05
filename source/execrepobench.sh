export LC_ALL="POSIX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python execrepobench.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-0.5B \
    --input_file data/exec_repo_bench.jsonl \
    --model_output_dir model_outputs/qwen25-05-base \
    --eval_output_dir eval_outputs/qwen25-05-base \
    --max_seq_length 8096 \
    --max_model_length 32768 \
    --gen_length 100 \
    --cfc_seq_length 2048 \
    --right_context_length 2048 \
    --tp 1 \
    --vram_utilization 0.9 \
    --workers 8 \
    --repo_dir /root/workspace/ExecRepoBench/repos \
    --env_path /root/workspace/ExecRepoBench/envs/envs
    # --lora_path /home/jovyan/anh-uet/models/loras/qwen25-coder-15-line-new
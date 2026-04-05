export LC_ALL="POSIX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python humaneval-fim-pass.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-0.5B \
    --input_dir data/HumanEval \
    --model_output_dir model_outputs/qwen25-05-base \
    --eval_output_dir eval_outputs/qwen25-05-base \
    --tasks single-line multi-line span span-light \
    --max_seq_length 4096 \
    --max_model_length 5000 \
    --workers 32 \
    --tp 1 \
    --vram_utilization 0.9 \
    # --lora_path /home/jovyan/anh-uet/models/loras/qwen25-15i-psm-line-tpu
    
    
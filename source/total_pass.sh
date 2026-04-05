#!/bin/bash
FINAL_OUTPUT_DIR=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_NAME=$3
GPU_INDEX=$4
VRAM_UTILIZATION=$5
LORA_PATH=$6
WORKER=16

export LC_ALL="POSIX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_INDEX}

MODEL_OUTPUT_DIR="model_outputs/${OUTPUT_NAME}"
EVAL_OUTPUT_DIR="eval_outputs/${OUTPUT_NAME}"

CMD_HUMAN="python humaneval-fim-pass.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --input_dir data/HumanEval \
    --model_output_dir ${MODEL_OUTPUT_DIR} \
    --eval_output_dir ${EVAL_OUTPUT_DIR} \
    --tasks single-line multi-line span span-light \
    --max_seq_length 4096 \
    --max_model_length 5000 \
    --workers ${WORKER} \
    --tp 1 \
    --vram_utilization ${VRAM_UTILIZATION}"
if [ ! -z "$LORA_PATH" ]; then
    CMD_HUMAN="${CMD_HUMAN} --lora_path ${LORA_PATH}"
fi

CMD_EXEC="python execrepobench.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --input_file data/exec_repo_bench.jsonl \
    --model_output_dir ${MODEL_OUTPUT_DIR} \
    --eval_output_dir ${EVAL_OUTPUT_DIR} \
    --max_seq_length 8096 \
    --max_model_length 32768 \
    --gen_length 100 \
    --cfc_seq_length 2048 \
    --right_context_length 2048 \
    --tp 1 \
    --vram_utilization ${VRAM_UTILIZATION} \
    --workers ${WORKER} \
    --repo_dir /root/workspace/ExecRepoBench/repos \
    --env_path /root/workspace/ExecRepoBench/envs/envs"
if [ ! -z "$LORA_PATH" ]; then
    CMD_EXEC="${CMD_EXEC} --lora_path ${LORA_PATH}"
fi

echo "Running command:"
echo $CMD_HUMAN
eval $CMD_HUMAN

echo "Running command:"
echo $CMD_EXEC
eval $CMD_EXEC

echo "Copying results to ${FINAL_OUTPUT_DIR}..."

# Create final output directory if it doesn't exist
mkdir -p "${FINAL_OUTPUT_DIR}"

# Copy everything inside eval output dir
cp -r "${EVAL_OUTPUT_DIR}/." "${FINAL_OUTPUT_DIR}/"

echo "All results copied to ${FINAL_OUTPUT_DIR}"

#!/bin/bash
FINAL_OUTPUT_DIR=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_NAME=$3
GPU_INDEX=$4
VRAM_UTILIZATION=$5
LORA_PATH=$6

export LC_ALL="POSIX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_INDEX}

MODEL_OUTPUT_DIR="model_outputs/${OUTPUT_NAME}"
EVAL_OUTPUT_DIR="eval_outputs/${OUTPUT_NAME}"

CMD_HUMAN="python humaneval-fim-sim.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --input_dir data/HumanEval \
    --model_output_dir ${MODEL_OUTPUT_DIR} \
    --eval_output_dir ${EVAL_OUTPUT_DIR} \
    --max_seq_length 4096 \
    --max_model_length 8256 \
    --gen_length 64 \
    --right_context_length 512 \
    --tp 1 \
    --vram_utilization ${VRAM_UTILIZATION}"
if [ ! -z "$LORA_PATH" ]; then
    CMD_HUMAN="${CMD_HUMAN} --lora_path ${LORA_PATH}"
fi

CMD_CCEVAL="python cceval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --input_file data/cceval/LANGUAGE/line_completion_oracle_bm25.jsonl \
    --model_output_dir ${MODEL_OUTPUT_DIR} \
    --eval_output_dir ${EVAL_OUTPUT_DIR} \
    --languages python java csharp typescript \
    --ts_lib grammar/LANGUAGE-lang-parser.so \
    --max_seq_length 8192 \
    --gen_length 50 \
    --cfc_seq_length 2048 \
    --right_context_length 2048 \
    --tp 1 \
    --vram_utilization ${VRAM_UTILIZATION}"
if [ ! -z "$LORA_PATH" ]; then
    CMD_CCEVAL="${CMD_CCEVAL} --lora_path ${LORA_PATH}"
fi

CMD_CCLONGEVAL="python cclongeval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --input_file data/cclongeval/python_TASK_sparse_oracle.jsonl \
    --model_output_dir ${MODEL_OUTPUT_DIR} \
    --eval_output_dir ${EVAL_OUTPUT_DIR} \
    --tasks chunk_completion function_completion \
    --ts_lib grammar/LANGUAGE-lang-parser.so \
    --max_seq_length 8096 \
    --gen_length 50 \
    --cfc_seq_length 2048 \
    --right_context_length 2048 \
    --tp 1 \
    --vram_utilization ${VRAM_UTILIZATION}"
if [ ! -z "$LORA_PATH" ]; then
    CMD_CCLONGEVAL="${CMD_CCLONGEVAL} --lora_path ${LORA_PATH}"
fi

CMD_REPOEVAL="python repoeval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --input_file data/repoeval/python_TASK_sparse_oracle.jsonl \
    --model_output_dir ${MODEL_OUTPUT_DIR} \
    --eval_output_dir ${EVAL_OUTPUT_DIR} \
    --tasks line_completion function_completion api_completion \
    --ts_lib grammar/LANGUAGE-lang-parser.so \
    --max_seq_length 8096 \
    --gen_length 50 \
    --cfc_seq_length 2048 \
    --right_context_length 2048 \
    --tp 1 \
    --vram_utilization ${VRAM_UTILIZATION}"
if [ ! -z "$LORA_PATH" ]; then
    CMD_REPOEVAL="${CMD_REPOEVAL} --lora_path ${LORA_PATH}"
fi

echo "Running command:"
echo $CMD_HUMAN
eval $CMD_HUMAN

echo "Running command:"
echo $CMD_CCEVAL
eval $CMD_CCEVAL

echo "Running command:"
echo $CMD_CCLONGEVAL
eval $CMD_CCLONGEVAL

echo "Running command:"
echo $CMD_REPOEVAL
eval $CMD_REPOEVAL

echo "Copying results to ${FINAL_OUTPUT_DIR}..."

# Create final output directory if it doesn't exist
mkdir -p "${FINAL_OUTPUT_DIR}"

# Copy everything inside eval output dir
cp -r "${EVAL_OUTPUT_DIR}/." "${FINAL_OUTPUT_DIR}/"

echo "All results copied to ${FINAL_OUTPUT_DIR}"

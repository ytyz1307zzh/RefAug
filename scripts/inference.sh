MODEL_DIR=$1  # path to model checkpoint
TEST_DATA_PATH=$2  # path to test.json file
OUTPUT_PATH=$3  # the path to save the output
NUM_GPUS=8

# TODO: Modify the inference script to directly use vLLM's parallel inference instead of manually splitting data
python src/model/split_input_data.py \
    -data ${TEST_DATA_PATH} \
    -n ${NUM_GPUS} \
    -output_dir ${MODEL_DIR} \
    -output_prefix test

for k in $(seq 0 $((${NUM_GPUS}-1))); do
    export CUDA_VISIBLE_DEVICES="${k}"

    accelerate launch \
        --num_machines 1 \
        --num_processes 1 \
        src/model/inference.py \
        --model ${MODEL_DIR} \
        --batch_size 1 \
        --data ${MODEL_DIR}/test_part_${k}.json \
        --output_path ${MODEL_DIR}/test-greedy-output_part_${k}.json \
        --precision bf16 \
        --max_input_length 1024 \
        --max_new_tokens 4096 \
        --temperature 0.0 \
        --seed 42 \
        --repetition_penalty 1.0 \
        --continue_output \
        --use_vllm &

done

wait

python src/model/merge_output_data.py \
    -input_prefix ${MODEL_DIR}/test-greedy-output \
    -n ${NUM_GPUS} \
    -output ${OUTPUT_PATH}

rm ${MODEL_DIR}/test_part_*.json
rm ${MODEL_DIR}/test-greedy-output_part_*.json

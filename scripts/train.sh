NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
GRADIENT_ACC_STEPS=16
PEAK_LR=1e-5
FINAL_LR=2e-6  # if set, scheduler must use linear decay
SEQ_LENGTH=4096
WEIGHT_DECAY=0.0
NUM_EPOCHS=3
WARMUP_RATIO=0.03
LOSS=sum  # sum loss or mean loss
SCHEDULE=linear
SEED=42
TRAIN_DIR=$1  # path to a directory that contains train.json and test.json

# Count the total batch size
TOTAL_BATCH_SIZE=$((${BATCH_SIZE_PER_GPU}*${NUM_GPUS}*${GRADIENT_ACC_STEPS}))

# directory of data and checkpoints
DATA_DIR=./data

CKPT_NAME=${TRAIN_DIR}/mistral-epoch${NUM_EPOCHS}-lr${PEAK_LR}-bs${TOTAL_BATCH_SIZE}-${LOSS}loss-seq${SEQ_LENGTH}-warm${WARMUP_RATIO}-${SCHEDULE}decay${WEIGHT_DECAY}-seed${SEED}

mkdir -p ${DATA_DIR}/${CKPT_NAME}
SCRIPT_PATH="$(readlink -f "$0")"
cp ${SCRIPT_PATH} ${DATA_DIR}/${CKPT_NAME}/train_script.sh

wandb login YOUR_WANDB_KEY

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file scripts/ds_config/zero3_no_offloading_accelerate.conf \
    --zero3_init_flag true \
    src/model/train.py \
    --train_file ${DATA_DIR}/${TRAIN_DIR}/train.json \
    --valid_file ${DATA_DIR}/${TRAIN_DIR}/train.json \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_seq_length ${SEQ_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --peak_learning_rate ${PEAK_LR} \
    --final_learning_rate ${FINAL_LR} \
    --lr_scheduler_type ${SCHEDULE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --reduce_loss ${LOSS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_checkpointing \
    --use_flash_attn \
    --output_dir ${DATA_DIR}/${CKPT_NAME} \
    --preprocessing_num_workers 4 \
    --checkpointing_steps epoch \
    --save_at_epoch "3" \
    --logging_steps 1 \
    --with_tracking \
    --report_to wandb \
    --wandb_username YOUR_WANDB_USERNAME \
    --seed ${SEED}

# Inference with vLLM
TEST_DATA_PATH=${DATA_DIR}/${TRAIN_DIR}/test.json

for i in 3;
do
    OUTPUT_DIR=${DATA_DIR}/${CKPT_NAME}/epoch_${i}

    # test set
    python src/model/split_input_data.py \
        -data ${TEST_DATA_PATH} \
        -n ${NUM_GPUS} \
        -output_dir ${OUTPUT_DIR} \
        -output_prefix test

    for k in $(seq 0 $((${NUM_GPUS}-1))); do
        export CUDA_VISIBLE_DEVICES="${k}"

        accelerate launch \
            --num_machines 1 \
            --num_processes 1 \
            src/model/inference.py \
            --model ${OUTPUT_DIR} \
            --batch_size 1 \
            --data ${OUTPUT_DIR}/test_part_${k}.json \
            --output_path ${OUTPUT_DIR}/test-greedy-output_part_${k}.json \
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
        -input_prefix ${OUTPUT_DIR}/test-greedy-output \
        -n ${NUM_GPUS} \
        -output ${OUTPUT_DIR}/test-greedy-output.json

    rm ${OUTPUT_DIR}/test_part_*.json
    rm ${OUTPUT_DIR}/test-greedy-output_part_*.json

done

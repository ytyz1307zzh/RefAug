
import os
import pdb
import json
import string
import argparse
from copy import deepcopy
from statistics import mean

import sys
sys.path.append("src/evaluate")
from eval_mathinstruct import answer_clean, number_it, compare_two_numbers

MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Follow the instruction to complete the request."
    },
    {
        "role": "user",
        "content": None
    }
]
INFERENCE_COMMAND = "bash scripts/inference.sh $MODEL_DIR$ $TEST_DATA_PATH$ $OUTPUT_PATH$"
EXTRACT_ANSWER_COMMAND = "python src/evaluate/remove_followup.py -data $DATA_PATH$ -output $OUTPUT_PATH$"


def eval_accuracy(pred_answer, ground_truth):
    ground_truth = number_it(ground_truth)
    assert ground_truth is not None
    if isinstance(pred_answer, str):
        pred_answer = number_it(pred_answer)
    if pred_answer is not None and compare_two_numbers(pred_answer, ground_truth):
        accuracy = True
    else:
        accuracy = False

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='data/original/test-multiturn-error-correct.json')
    parser.add_argument("-output", type=str, default="CHECKPOINT_DIR/test-multiturn-error-correct.json")
    parser.add_argument("-extract_answer", action="store_true", help='If True, we need to extract the answer part and discard the follow-up parts from the output prediction.')
    parser.add_argument("-format", choices=['prompt_completion', 'chat_messages'], default='chat_messages', help='Whether to use prompt completion format or chat messages format when running model inference')
    args = parser.parse_args()

    data = json.load(open(args.data, 'r', encoding='utf8'))
    print(f"Loaded {len(data)} from {args.data}")

    # Extract the directory name from args.output
    # args.output = XXX/data_dir/ckpt_name/epoch_3/multiturn-error-correct/test-multiturn-error-correct.json
    paths = args.output.split("/")
    model_dir = "/".join(paths[:-2])
    output_dir = "/".join(paths[:-1])
    base_filename = paths[-1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Process input data
    id2answer = {}
    test_data = []

    for example in data:
        id_ = example['id']
        question = example['question']
        answer = example['answer']

        id2answer[id_] = answer

        if args.format == 'chat_messages':
            messages = deepcopy(MESSAGE_TEMPLATE)
            messages[1]['content'] = question

            test_data.append({
                "id": id_,
                "question": question,
                "answer": answer,
                "messages": messages
            })
        elif args.format == "prompt_completion":
            test_data.append({
                "id": id_,
                "question": question,
                "answer": answer,
                "prompt": question
            })

        else:
            raise ValueError(f"Invalid format: {args.format}")

    input_file_name = base_filename.replace(".json", "-input.json")
    input_path = os.path.join(output_dir, input_file_name)
    json.dump(test_data, open(input_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print(f"Saved {len(test_data)} examples to {input_path}")

    # Run inference with model
    output_file_name = base_filename.replace(".json", "-output.json")
    output_path = os.path.join(output_dir, output_file_name)

    if os.path.exists(output_path):
        print(f"Skip inference because {output_path} already exists! Directly reading the old results...")
    else:
        inference_command = INFERENCE_COMMAND.replace("$MODEL_DIR$", model_dir).replace("$TEST_DATA_PATH$", input_path).replace("$OUTPUT_PATH$", output_path)

        os.system(inference_command)

    # Evaluate model responses
    if args.extract_answer:
        extract_answer_command = EXTRACT_ANSWER_COMMAND.replace("$DATA_PATH$", output_path)
        output_path = output_path.replace(".json", "-extracted.json")
        extract_answer_command = extract_answer_command.replace("$OUTPUT_PATH$", output_path)
        os.system(extract_answer_command)

    predicted_data = json.load(open(output_path, 'r', encoding='utf8'))

    # Process model response and calculate answer accuracy for each test example
    unfinished_answer_cnt = 0
    results = []
    for example in predicted_data:
        id_ = example['id']
        model_response = example['output']

        if len(model_response) > 0 and "The answer is" not in model_response and model_response.strip()[-1] not in string.punctuation:
                unfinished_answer_cnt += 1

        ground_truth = id2answer[id_]
        pred_answer = answer_clean("gsm8k", 'The answer is', model_response)

        accuracy = eval_accuracy(pred_answer=pred_answer, ground_truth=ground_truth)

        results.append({
            "id": id_,
            "question": example['instruction'],
            "answer": ground_truth,
            "output": model_response,
            "pred_answer": pred_answer,
            "accuracy": accuracy
        })
    
    eval_output_path = output_dir + "/eval-" + output_path.split('/')[-1]
    json.dump(results, open(eval_output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print(f"Saved {len(results)} examples to {eval_output_path}")

    # Calculate the accuracy
    mean_accuracy = mean([ex['accuracy'] for ex in results])
    print(f"Accuracy ({len(results)} examples): {mean_accuracy:.2%}")
    print(f"Unfinished answer count: {unfinished_answer_cnt}")


if __name__ == "__main__":
    main()

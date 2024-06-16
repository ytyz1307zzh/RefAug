
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
    parser.add_argument("-data", type=str, default='data/original/test-multiturn-followup.json')
    parser.add_argument("-output", type=str, default="CHECKPOINT_DIR/test-multiturn-followup.json")
    parser.add_argument("-extract_answer", action="store_true", help='If True, we need to extract the answer part and discard the follow-up parts from the output prediction.')
    parser.add_argument("-format", choices=['prompt_completion', 'chat_messages'], default='chat_messages', help='Whether to use prompt completion format or chat messages format when running model inference')
    args = parser.parse_args()

    data = json.load(open(args.data, 'r', encoding='utf8'))
    print(f"Loaded {len(data)} from {args.data}")

    # Extract the directory name from args.output
    # args.output = XXX/data_dir/ckpt_name/epoch_3/multiturn-followup/test-multiturn-followup.json
    paths = args.output.split("/")
    model_dir = "/".join(paths[:-2])
    output_dir = "/".join(paths[:-1])
    base_filename = paths[-1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Iterate 3 times for multi-turn generation
    num_turns = 3
    results = {f"turn_{k}": [] for k in range(1, num_turns + 1)}
    id2data = {ex['id']: {"input_text": "", "model_response": "", "answer": None} for ex in data}
    id2accuracy = {ex['id']: [] for ex in data}  # Accuracy for each turn on the same example

    for i in range(1, num_turns + 1):
        print("-" * 20 + f" Turn {i} " + "-" * 20)
        
        # Process input data for this turn
        test_data_this_turn = []
        for example in data:

            id_ = example['id']
            question_this_turn = example[f'Q{i}']
            answer_this_turn = example[f'A{i}']

            input_text_last_turn = id2data[id_]["input_text"]
            model_response_last_turn = id2data[id_]["model_response"]

            input_text_this_turn = input_text_last_turn + "\n\n" + model_response_last_turn + '\n\n' + question_this_turn
            input_text_this_turn = input_text_this_turn.strip()

            if args.format == "chat_messages":
                messages = deepcopy(MESSAGE_TEMPLATE)
                messages[1]['content'] = input_text_this_turn

                test_data_this_turn.append({
                    "id": id_,
                    "question": input_text_this_turn,
                    "answer": answer_this_turn,
                    "messages": messages
                })

            elif args.format == "prompt_completion":
                test_data_this_turn.append({
                    "id": id_,
                    "question": input_text_this_turn,
                    "answer": answer_this_turn,
                    "prompt": input_text_this_turn
                })

            else:
                raise ValueError(f"Invalid format: {args.format}")

            id2data[id_]["input_text"] = input_text_this_turn
            id2data[id_]["answer"] = answer_this_turn
        
        input_file_name = base_filename.replace(".json", f"-turn_{i}-input.json")
        input_path = os.path.join(output_dir, input_file_name)
        json.dump(test_data_this_turn, open(input_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
        print(f"Saved {len(test_data_this_turn)} examples to {input_path}")

        # Run inference
        output_file_name = base_filename.replace(".json", f"-turn_{i}-output.json")
        output_path = os.path.join(output_dir, output_file_name)

        if os.path.exists(output_path):
            print(f"Skip inference for turn {i} because {output_path} already exists! Directly reading the old results...")
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
        id2pred = {ex['id']: ex['output'] for ex in predicted_data}

        # Process model response and calculate answer accuracy for each test example
        unfinished_answer_cnt = 0
        for example in data:
            id_ = example['id']
            model_response = id2pred[id_]
            id2data[id_]["model_response"] = model_response

            if len(model_response) > 0 and "The answer is" not in model_response and model_response.strip()[-1] not in string.punctuation:
                unfinished_answer_cnt += 1
            
            ground_truth = id2data[id_]["answer"]
            pred_answer = answer_clean("gsm8k", 'The answer is', model_response)

            accuracy = eval_accuracy(pred_answer=pred_answer, ground_truth=ground_truth)

            results[f"turn_{i}"].append({
                "id": id_,
                "question": id2data[id_]["input_text"],
                "answer": ground_truth,
                "output": model_response,
                "pred_answer": pred_answer,
                "accuracy": accuracy
            })

            id2accuracy[id_].append(accuracy)

        eval_output_path = output_dir + "/eval-" + output_path.split('/')[-1]
        json.dump(results[f"turn_{i}"], open(eval_output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
        print(f"Saved {len(results[f'turn_{i}'])} examples to {eval_output_path}")

        # Calculate the accuracy of this turn
        accuracy_this_turn = mean([ex['accuracy'] for ex in results[f"turn_{i}"]])
        print(f"Turn {i} accuracy ({len(results[f'turn_{i}'])} examples): {accuracy_this_turn:.2%}")
        print(f"Unfinished answer count: {unfinished_answer_cnt}")

    # Check the accuracy difference across different turns
    all_correct = sum([all(acc) for acc in id2accuracy.values()])
    all_wrong = sum([not any(acc) for acc in id2accuracy.values()])
    first_correct_but_second_wrong = sum([acc[0] and not acc[1] for acc in id2accuracy.values()])
    first_second_correct_but_third_wrong = sum([acc[0] and acc[1] and not acc[2] for acc in id2accuracy.values()])
    first_wrong_but_second_correct = sum([not acc[0] and acc[1] for acc in id2accuracy.values()])
    first_wrong_but_third_correct = sum([not acc[0] and acc[2] for acc in id2accuracy.values()])
    second_wrong_but_third_correct = sum([not acc[1] and acc[2] for acc in id2accuracy.values()])
    print(f'All correct: {all_correct / len(id2accuracy):.2%}')
    print(f'All wrong: {all_wrong / len(id2accuracy):.2%}')
    print(f'First correct but second wrong: {first_correct_but_second_wrong / len(id2accuracy):.2%}')
    print(f'First & second correct but third wrong: {first_second_correct_but_third_wrong / len(id2accuracy):.2%}')
    print(f'First wrong but second correct: {first_wrong_but_second_correct / len(id2accuracy):.2%}')
    print(f'First wrong but third correct: {first_wrong_but_third_correct / len(id2accuracy):.2%}')
    print(f'Second wrong but third correct: {second_wrong_but_third_correct / len(id2accuracy):.2%}')


if __name__ == "__main__":
    main()

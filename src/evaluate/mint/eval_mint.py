import re
import os
import pdb
import time
import json
import openai
import random
import shutil
import argparse
from copy import deepcopy
from statistics import mean

import sys
sys.path.append("src")
from evaluate.eval_mathinstruct import answer_clean, number_it, compare_two_numbers
from evaluate.mint.prompt_template import ANSWER_MATCH_TEMPLATE, EXPERT_FEEDBACK_TEMPLATE
from tqdm import tqdm

random.seed(42)

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


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    print(f'Saved {len(data)} data to {path}')


def remove_repeated_ending(text):
    """
    If the answer has a repetitive ending, remove it.
    """
    text_length = len(text)
    
    for n in range(1, text_length // 6):
        n_gram = text[-n:]
        count = 1
        for i in range(text_length - n, -1, -n):
            if text[i - n:i] == n_gram:
                count += 1
            else:
                break
        
        if count > 5:
            return text[:-n * (count - 1)]
    
    return text


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


def parallel_openai_request(
    args,
    all_messages,
    model,
    max_tokens,
    all_ids,
    request_save_path,
    response_save_path
):
    all_requests = []
    finished_ids = []

    if os.path.exists(response_save_path):
        finished_responses = read_jsonl_as_list(response_save_path)
        finished_ids = [x[2]['id'] for x in finished_responses]

        # If there are still unfinished requests, save the previous requests (only for logging purposes, no impact on the current run)
        # if len(finished_ids) < len(all_ids):
        #     shutil.copy(request_save_path, request_save_path.replace(".jsonl", "-old.jsonl"))
        # The response data are appended to response_save_path during OpenAI requests so no need to backup

        print(f'Skipping {len(finished_ids)} because we find their results in a previous run...')

    for messages, id_ in zip(all_messages, all_ids):
        if id_ in finished_ids:
            continue

        request_json = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 0,
            "metadata": {"id": id_}
        }
        all_requests.append(request_json)

    if len(all_requests) > 0:
        save_list_as_jsonl(request_save_path, all_requests)

        print('*' * 15 + "Start requesting OpenAI" + '*' * 15)
        start_time = time.time()

        openai_request_command = f"python src/utils/openai_parallel_request.py" \
                                    f" --requests_filepath {request_save_path}" \
                                    f" --save_filepath {response_save_path}" \
                                    f" --request_url {args.url}" \
                                    f" --api_key {args.api_key}" \
                                    f" --max_requests_per_minute {args.max_rpm}" \
                                    f" --max_tokens_per_minute {args.max_tpm}" \
                                    f" --token_encoding_name {args.tokenizer}" \
                                    f" --max_attempts {args.max_attempts}"
        
        os.system(openai_request_command)

        print('*' * 15 + "Finished requesting OpenAI" + '*' * 15)
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    else:
        print("Skip requesting OpenAI because we found old results! Directly reading the old results...")

    response_data = read_jsonl_as_list(response_save_path)

    id2response = {}
    for example in response_data:
        try:
            id2response[example[2]["id"]] = example[1]["choices"][0]["message"]["content"]
        except TypeError:
            try:
                if len(example[1]) == args.max_attempts and \
                        eval(example[1][-1])["error"]["code"] in ["context_length_exceeded", "invalid_prompt"]:
                    print(f'OpenAI Response Error {eval(example[1][-1])["error"]["code"]} in {example[2]["id"]}')
                else:
                    print(f'Other Error in {example[2]["id"]}')
            except SyntaxError:
                print(f'Other Error in {example[2]["id"]} (probably content filtering, or rate limit exceeded)')
            
            id2response[example[2]["id"]] = "N/A"
            
    return id2response


class State:
    """
    The state of a current problem-solving agent.
    """

    def __init__(
        self,
        example
    ) -> None:
        self.id = example['id']

        assert len(example['messages']) == 2
        # All conversation history (inputs, outputs, feedbacks, etc)
        self.history = [example['messages'][1]]

        # The string form of input in the current turn
        self.current_input = example['messages'][1]['content']

        self.question = self.current_input.split('\n\nTask:\n')[-1].strip()
        self.answer = example['answer']

        if isinstance(self.answer, float):
            self.answer = round(self.answer, 2)
        self.answer = str(self.answer)

        self.answer_type = example['answer_type']

        self.success = False
        self.success_turn = -1

    def __str__(self):
        return 'Task:\n' + self.current_input.split('\n\nTask:\n')[-1].strip()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='test-mint-original-prompt.json')
    parser.add_argument("-output", type=str, default="CHECKPOINT_DIR/test-mint.json")
    parser.add_argument("-extract_answer", action="store_true", help='If True, we need to extract the answer part and discard the follow-up parts from the output prediction.')
    parser.add_argument('-max_turns', type=int, default=5, help='How many iterations does the assistant have to solve the task?')
    parser.add_argument("-feedback_with_gt", action="store_true", help='If True, we will provide the feedback agent with ground truth answer.')
    parser.add_argument("-answer_extract_agent", type=str, default=
    'gpt-4-0125-preview')
    parser.add_argument("-feedback_agent", type=str, default='gpt-4-0613')
    parser.add_argument("-url", type=str, default="https://api.openai.com/v1/chat/completions", help='URL to OpenAI endpoint')
    parser.add_argument("-api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("-max_tpm", type=int, default=100000, help="Maximum tokens per minute")
    parser.add_argument("-max_rpm", type=int, default=500, help="Maximum requests per minute")
    parser.add_argument("-tokenizer", type=str, default="cl100k_base", help='tiktoken tokenizer')
    parser.add_argument("-max_attempts", type=int, default=10, help="Max number of retries if failed")
    args = parser.parse_args()
    openai.api_key = args.api_key

    data = json.load(open(args.data, 'r', encoding='utf8'))
    print(f"Loaded {len(data)} from {args.data}")

    # Extract the directory name from args.output
    # args.output = XXX/data_dir/ckpt_name/epoch_3/mint/test-mint.json
    paths = args.output.split("/")
    model_dir = "/".join(paths[:-2])
    output_dir = "/".join(paths[:-1])
    base_filename = paths[-1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize all states
    states = [State(example) for example in data]
    id2state = {state.id: state for state in states}

    for i in range(1, args.max_turns + 1):
        print('\n' + '-' * 15 + f'Turn {i}' + '-' * 15 + '\n')

        # Collect input data to the LLM
        test_data_this_turn = []

        for state in states:

            if state.success:
                continue

            message = deepcopy(MESSAGE_TEMPLATE)
            message[1]['content'] = state.current_input

            test_data_this_turn.append({
                "id": state.id,
                "question": state.question,
                "answer": state.answer,
                "messages": message
            })

        if len(test_data_this_turn) == 0:
            break

        input_file_name = base_filename.replace(".json", f"-turn_{i}-input.json")
        input_path = os.path.join(output_dir, input_file_name)
        json.dump(test_data_this_turn, open(input_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
        print(f"Saved {len(test_data_this_turn)} examples to {input_path}")

        output_file_name = base_filename.replace(".json", f"-turn_{i}-output.json")
        output_path = os.path.join(output_dir, output_file_name)

        # Run LLM inference
        if os.path.exists(output_path):
            print(f"Skip inference for turn {i} because {output_path} already exists! Directly reading the old results...")
        else:
            inference_command = INFERENCE_COMMAND.replace("$MODEL_DIR$", model_dir).replace("$TEST_DATA_PATH$", input_path).replace("$OUTPUT_PATH$", output_path)

            inference_command += f" > /dev/null"

            print(f'\nRunning inference command: {inference_command}\n')

            os.system(inference_command)

        # ---------- Post-process the generated response ----------
        # Extract the answer and remove the reflection part, if necessary
        if args.extract_answer:
            print('\n' + '-' * 15 + 'Remove follow-up' + '-' * 15 + '\n')
            extract_answer_command = EXTRACT_ANSWER_COMMAND.replace("$DATA_PATH$", output_path)
            output_path = output_path.replace(".json", "-extracted.json")
            extract_answer_command = extract_answer_command.replace("$OUTPUT_PATH$", output_path)
            os.system(extract_answer_command)

        predicted_data = json.load(open(output_path, 'r', encoding='utf8'))
        id2pred = {ex['id']: remove_repeated_ending(ex['output']) for ex in predicted_data}

        # Answer evaluation using LLM: prepare prompts
        print('\n' + '-' * 15 + 'Answer extraction' + '-' * 15 + '\n')
        answer_extract_messages, answer_extract_ids = [], []
        for id_, prediction in id2pred.items():
            state = id2state[id_]
            question = state.question
            answer = state.answer

            # Use helper functions from eval_mathinstruct.py to evaluate gsm and math
            if id_.startswith("gsm8k") or id_.startswith("math"):
                continue

            answer_extract_prompt = deepcopy(ANSWER_MATCH_TEMPLATE)
            answer_extract_prompt = answer_extract_prompt.replace("$$QUESTION$$", question).replace("$$ANSWER$$", answer).replace("$$PREDICTION$$", prediction)

            messages = [
                {"role": "system", "content": "You are a helpful assistant and good at math."},
                {"role": "user", "content": answer_extract_prompt},
            ]

            answer_extract_messages.append(messages)
            answer_extract_ids.append(id_)

        # Answer evaluation using LLM: call ChatGPT
        id2answer_extract_response = parallel_openai_request(
            args,
            all_messages=answer_extract_messages,
            model=args.answer_extract_agent,
            max_tokens=128,
            all_ids=answer_extract_ids,
            request_save_path=output_dir + "/temp-save-" + output_path.split('/')[-1].replace(".json", "-answer-extract-requests.jsonl"),
            response_save_path=output_dir + "/temp-save-" + output_path.split('/')[-1].replace(".json", "-answer-extract-response.jsonl")
        )

        # Answer evaluation using LLM: check task success & prepare feedback prompts
        print('\n' + '-' * 15 + 'Check success' + '-' * 15 + '\n')
        success_cnt, fail_cnt = 0, 0
        results_this_turn = []
        feedback_messages, feedback_ids = [], []

        for id_, prediction in id2pred.items():
            state = id2state[id_]
            question = state.question
            answer = state.answer

            # Append the prediction to state
            state.current_input += f"\n\nAssistant:\n{prediction}"
            state.history.append({"role": "assistant", "content": f"Assistant:\n{prediction}"})

            if id_.startswith("gsm8k"):
                pred_answer = answer_clean("gsm8k", 'The answer is', prediction)
                if eval_accuracy(ground_truth=answer, pred_answer=pred_answer):
                    judgment = "correct"
                    answer_extract_response = f"The model's answer is {pred_answer}. The reference answer is {answer}. Therefore, the answer is \\boxed{{correct}}."
                else:
                    judgment = "wrong"
                    answer_extract_response = f"The model's answer is {pred_answer}. The reference answer is {answer}. Therefore, the answer is \\boxed{{wrong}}."

                id2answer_extract_response[id_] = answer_extract_response
            
            # MINT's split of MATH dataset only contains numeric answer
            elif id_.startswith("math"):
                pred_answer = answer_clean("math", 'The answer is', prediction)
                if eval_accuracy(ground_truth=answer, pred_answer=pred_answer):
                    judgment = "correct"
                    answer_extract_response = f"The model's answer is {pred_answer}. The reference answer is {answer}. Therefore, the answer is \\boxed{{correct}}."
                else:
                    judgment = "wrong"
                    answer_extract_response = f"The model's answer is {pred_answer}. The reference answer is {answer}. Therefore, the answer is \\boxed{{wrong}}."
                    
                id2answer_extract_response[id_] = answer_extract_response

            else:
                answer_extract_response = id2answer_extract_response[id_]

                # Parse LLM judgment
                try:
                    judgment = re.findall(r"\\boxed{(.+?)}", answer_extract_response)[-1]
                    assert judgment == "correct" or judgment == "wrong"
                except (TypeError, IndexError, AssertionError) as error:
                    print(f"Error in GPT judgment: {error}")
                    print(f"GPT judgment: {answer_extract_response}")
                    judgment = "wrong"

            # If the answer is correct, feedback is not needed
            if judgment == "correct":
                state.success = True
                state.success_turn = i
                success_cnt += 1

                results_this_turn.append({
                    "id": id_,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "evaluation": answer_extract_response
                })

                continue

            fail_cnt += 1

            # If this is already the final turn, feedback is not needed
            if i == args.max_turns:
                results_this_turn.append({
                    "id": id_,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "evaluation": answer_extract_response,
                })
                continue

            # If the answer is incorrect, provide feedback
            observation = f"Your answer is wrong.\nYou have {args.max_turns - i} chances to propose solution left."

            state.current_input += f"\n\nObservation:\n{observation}"
            trajectory = "Task:\n" + state.current_input.split('\n\nTask:\n')[-1].strip()

            feedback_prompt = deepcopy(EXPERT_FEEDBACK_TEMPLATE)
            feedback_prompt = feedback_prompt.replace("$$TRAJECTORY$$", trajectory)

            if args.feedback_with_gt:
                feedback_prompt = feedback_prompt.replace("$$ANSWER$$", answer)
            else:
                feedback_prompt = feedback_prompt.replace("$$ANSWER$$", "NOT GIVEN")

            feedback_messages.append([
                {"role": "user", "content": feedback_prompt},
            ])
            feedback_ids.append(id_)

        # If this is already the final turn, feedback is not needed, break the loop from here
        if i == args.max_turns:
            print(f"Turn {i} success: {success_cnt}, fail: {fail_cnt}")
            eval_output_path = output_dir + "/eval-" + output_path.split('/')[-1]
            json.dump(results_this_turn, open(eval_output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
            break

        # Collect feedback: call ChatGPT
        print('\n' + '-' * 15 + 'Collect feedback' + '-' * 15 + '\n')
        id2feedback_response = parallel_openai_request(
            args,
            all_messages=feedback_messages,
            model=args.feedback_agent,
            max_tokens=512,
            all_ids=feedback_ids,
            request_save_path=output_dir + "/temp-save-" + output_path.split('/')[-1].replace(".json", "-feedback-requests.jsonl"),
            response_save_path=output_dir + "/temp-save-" + output_path.split('/')[-1].replace(".json", "-feedback-response.jsonl")
        )

        for id_, prediction in id2pred.items():
            state = id2state[id_]
            question = state.question
            answer = state.answer

            if state.success:
                continue

            feedback_response = id2feedback_response[id_]
            answer_extract_response = id2answer_extract_response[id_]

            observation = f"Your answer is wrong.\nYou have {args.max_turns - i} chances to propose solution left."

            state.current_input += f"\n\nExpert feedback:\n{feedback_response}"
            state.history.append({"role": "user", "content": f"Observation:\n{observation}\n\nExpert feedback:\n{feedback_response}"})

            results_this_turn.append({
                "id": id_,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "evaluation": answer_extract_response,
                "observation": observation,
                "feedback": feedback_response
            })

        print(f"Turn {i} success: {success_cnt}, fail: {fail_cnt}")
        eval_output_path = output_dir + "/eval-" + output_path.split('/')[-1]
        json.dump(results_this_turn, open(eval_output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)

    # Calculate success rate per turn
    turn2success = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for state in states:
        
        if not state.success:
            continue

        success_turn = state.success_turn
        for turn in turn2success.keys():
            if int(turn) >= success_turn:
                turn2success[turn] += 1

    print('-' * 30)
    for turn, success_cnt in turn2success.items():
        print(f'Turn {turn} accuracy: {success_cnt / len(data):.2%}')


if __name__ == "__main__":
    main()

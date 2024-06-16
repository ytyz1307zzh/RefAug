"""
Get the follow-up general problem for each original math word problem in the training set
"""

import os
import json
import time
import random
random.seed(42)
import shutil
import argparse

import sys
sys.path.append('src/data')
from prompt_template import (
    ANSWER_AUGMENT_TEMPLATE,
    QUESTION_AUGMENT_TEMPLATE,
    ALTERNATIVE_FOLLOWUP_REFLECT_TEMPLATE,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default="train-qa-format.json", help='A JSON file with all Q-A data. Questions should in a field called "question" and answers in a field called "response". Also need a "id" field to identify the instance.')
    parser.add_argument("-request_save_path", default="temp_save_request.jsonl", help='Temporary file to save the requests to OpenAI')
    parser.add_argument("-response_save_path", default="temp_save_response.jsonl", help='Temporary file to save the responses from OpenAI')
    parser.add_argument("-output_path", default="train-reflection.json", help='Output file to save the augmented data')
    parser.add_argument("-augment_type", default="reflection", choices=['reflection', 'answer-augment', 'question-augment'])
    parser.add_argument("-model", type=str, default="gpt-4-0125-preview")
    parser.add_argument("-max_tokens", type=int, default=512)
    parser.add_argument("-temperature", type=float, default=0.7)
    parser.add_argument("-top_p", type=float, default=1.0)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-url", type=str, default="https://api.openai.com/v1/chat/completions", help='URL to OpenAI endpoint')
    parser.add_argument("-api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("-max_tpm", type=int, default=100000, help="Maximum tokens per minute")
    parser.add_argument("-max_rpm", type=int, default=500, help="Maximum requests per minute")
    parser.add_argument("-tokenizer", type=str, default="cl100k_base", help='tiktoken tokenizer')
    parser.add_argument("-max_attempts", type=int, default=20, help="Max number of retries if failed")
    args = parser.parse_args()

    data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    print(f"Loaded {len(data)} data from {args.data_path}")
    id2example = {}

    final_results = []
    if os.path.exists(args.output_path):
        final_results = json.load(open(args.output_path, "r", encoding="utf-8"))
        # Find the IDs of the finished instances (skipping the prompt element)
        final_results = [ex for ex in final_results if "id" in ex.keys()]
        finished_ids = {ex["id"] for ex in final_results}
        print(f'Found previous finished {len(finished_ids)} instances!')
        data = [ex for ex in data if ex["id"] not in finished_ids]
        print(f"Remaining {len(data)} instances to process!")

    require_response = True
    if args.augment_type == "answer-augment":
        prompt_template = ANSWER_AUGMENT_TEMPLATE
        require_response = False
    elif args.augment_type == "question-augment":
        prompt_template = QUESTION_AUGMENT_TEMPLATE
        require_response = False
    elif args.augment_type == "reflection":
        prompt_template = ALTERNATIVE_FOLLOWUP_REFLECT_TEMPLATE
    else:
        raise ValueError(f"Invalid augment type: {args.augment_type}")

    all_requests = []
    for example in data:
        id_ = example['id']
        question = example['question']

        if require_response:
            response = example['response']
        else:
            assert "$$RESPONSE$$" not in prompt_template
            response = ""

        id2example[id_] = {
            "question": question,
            "response": response
        }

        openai_prompt = prompt_template.replace("$$QUESTION$$", question).replace("$$RESPONSE$$", response)
        system_prompt = "You are a helpful assistant and good at math."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_prompt},
        ]

        request_json = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "metadata": {"id": id_}
        }
        all_requests.append(request_json)

    save_list_as_jsonl(args.request_save_path, all_requests)

    if not os.path.exists(args.response_save_path):

        print('*' * 15 + "Start requesting OpenAI" + '*' * 15)
        start_time = time.time()

        openai_request_command = f"python src/utils/openai_parallel_request.py" \
                                    f" --requests_filepath {args.request_save_path}" \
                                    f" --save_filepath {args.response_save_path}" \
                                    f" --request_url {args.url}" \
                                    f" --api_key {args.api_key}" \
                                    f" --max_requests_per_minute {args.max_rpm}" \
                                    f" --max_tokens_per_minute {args.max_tpm}" \
                                    f" --token_encoding_name {args.tokenizer}" \
                                    f" --max_attempts {args.max_attempts}"
        
        os.system(openai_request_command)

        print('*' * 15 + "Finished requesting OpenAI" + '*' * 15)
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    openai_generate_data = read_jsonl_as_list(args.response_save_path)

    skip_cnt = 0
    for example in openai_generate_data:
        # example[0]: input (except metadata)
        # example[1]: OpenAI response
        # example[2]: metadata
        id_ = example[2]["id"]
        original_example = id2example[id_]

        try:
            final_results.append({
                "id": id_,
                "question": original_example['question'],
                "response": original_example['response'],
                "output": example[1]["choices"][0]["message"]["content"]
            })
        except:
            skip_cnt += 1
            continue

    final_results = sorted(final_results, key=lambda x: x['id'])
    if os.path.exists(args.output_path):
        shutil.copy(args.output_path, args.output_path + '.bak')
        json.dump(final_results, open(args.output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
        os.remove(args.output_path + '.bak')
    else:
        json.dump(final_results, open(args.output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print(f'Saved {len(final_results)} examples to {args.output_path}')
    print(f'Skipped {skip_cnt} instances due to content filter or incomplete response')


if __name__ == "__main__":
    main()

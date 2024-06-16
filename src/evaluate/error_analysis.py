
import os
import re
import json
import time
import shutil
import argparse


PROMPT_TEMPLATE = """You will be given a math word problem, its reference solution, and a wrong solution from a student. Based on the reasoning path in the reference solution, please help me identify the error type in the wrong solution. In general, we consider two types of errors:

1. Calculation Error: The student makes mistakes in handling mathematical equations. This mainly include (1) incorrect analysis of quantitative relationships, and (2) numerical computation mistakes.
2. Reasoning Error: The student makes mistakes of carrying out the reasoning process. This can include multiple subtypes, such as missing a reasoning step, semantic misunderstanding of the problem, incoherent steps (some reasoning steps do not follow the previous ones), the loss of a condition, etc.

Please first write a brief analysis of the error, and then give your final judgment of the error type. The error type should be one of the three options: [[Calculation Error]], [[Reasoning Error]], or [[Both]] if there are both reasoning errors and calculation errors in the solution. Make sure to wrap your judgment in [[]].

Here are two examples:

# Example 1:

## Question:
Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

## Reference Solution:
The cost of the house and repairs came out to 80,000+50,000=$130,000\nHe increased the value of the house by 80,000*1.5=120,000\nSo the new value of the house is 120,000+80,000=$200,000\nSo he made a profit of 200,000-130,000=$70,000\nThe answer is 70000

## Wrong Solution:
The house was worth 80000*.15=$12,000 more than when he bought it\nSo he sold it for 80000+12000=$92,000\nHe made a profit of 92000-80000-50000=$22,000\nThe answer is 22000.

## Judgment:
The student made a error in identifying the quantitative relationship to calculate the increased value of the house. The student should have multiplied the original value of the house by 1.5, not 0.15. Besides, the student also made a computation mistake when calculating the profit. The operation 92000-80000-50000 does not equal to 22000. Both errors fall into the calculation error type. Therefore, the error type here is [[Calculation Error]].

# Example 2:

## Question:
Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?

## Reference Solution:
The discount price of one glass is 60/100 * 5 = $3.\nIf every second glass is cheaper, that means Kylar is going to buy 16 / 2 = 8 cheaper glasses.\nSo for the cheaper glasses, Kylar is going to pay 8 * 3 = $24.\nAnd for the regular-priced glasses, Kylar will pay 8 * 5 = $40.\nSo in total Kylar needs to pay 24 + 40 = $64 for the glasses he wants to buy.\nThe answer is 64

## Wrong Solution:
For every second glass, Kylar needs to pay 5 * 60/100 = $3.\nFor 16 glasses, he needs to pay 16 * 5 = $80.\nFor the next 8 glasses, the price will be 8 * 3 = $24.\nIn total Kylar needs to pay 80 + 24 = $104.\nThe answer is 104.

## Judgment:
The student misunderstood the question. The original question stated that for each pair of glasses, the first glass costs $5 and the second glass costs 60% of the price. Therefore, if Kylar buys 16 glasses, he will buy 8 pairs of glasses, of which 8 pieces are in the original price and the other 8 in the discounted price. The student mistakenly interpreted this into buying 16 glasses with the original price and another 8 glasses with the discounted price. This is a semantic misunderstanding of the problem, which falls into the reasoning error type. Therefore, the error type here is [[Reasoning Error]].

# Now solve the following example:

## Question:
$$QUESTION$$

## Reference Solution:
$$REFERENCE$$

## Wrong Solution:
$$RESPONSE$$

## Judgment:
"""


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
    parser.add_argument("-data_path", default="PATH/TO/gsm8k_test.json", help='Path to the original GSM8K test data. The data should be in JSON format and contain the fields "id", "question", and "rationale" (CoT).')
    parser.add_argument("-pred_path", default="test-greedy-output-extracted.json", help='Model predictions')
    parser.add_argument("-request_save_path", default="temp_save_request.jsonl")
    parser.add_argument("-response_save_path", default="temp_save_response.jsonl")
    parser.add_argument("-output_path", default="error-analysis.json")
    parser.add_argument("-model", type=str, default="gpt-4o")
    parser.add_argument("-max_tokens", type=int, default=512)
    parser.add_argument("-temperature", type=float, default=0)
    parser.add_argument("-top_p", type=float, default=0)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-url", type=str, default="https://api.openai.com/v1/chat/completions", help='URL to OpenAI endpoint')
    parser.add_argument("-api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("-max_tpm", type=int, default=100000, help="Maximum tokens per minute")
    parser.add_argument("-max_rpm", type=int, default=500, help="Maximum requests per minute")
    parser.add_argument("-tokenizer", type=str, default="cl100k_base", help='tiktoken tokenizer')
    parser.add_argument("-max_attempts", type=int, default=20, help="Max number of retries if failed")
    args = parser.parse_args()

    original_data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    id2rationale = {}

    for example in original_data:
        id_ = f"gsm8k-{example['id']}"
        question = example['question']
        rationale = example['rationale']

        # remove all text wrapped with <<>> in rationale
        rationale = re.sub(r'<<.+?>>', '', rationale)
        assert rationale.count("#### ") == 1
        # replace #### with "The answer is "
        rationale = re.sub(r'#### ', 'The answer is ', rationale)

        id2rationale[id_] = {"question": question, "rationale": rationale}

    print(f"Loaded {len(id2rationale)} original data from {args.data_path}")

    data = json.load(open(args.pred_path, 'r', encoding='utf-8'))
    data = [x for x in data if x['id'].startswith("gsm8k")]
    print(f'Loaded {len(data)} predictions from {args.pred_path}')
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

    all_requests = []
    for example in data:
        id_ = example['id']
        question = example['question']
        prediction = example['output']
        accuracy = example['accuracy']

        # Only check the incorrect predictions
        if accuracy:
            continue

        id2example[id_] = {
            "question": question,
            "prediction": prediction
        }
        assert question == id2rationale[id_]['question']
        rationale = id2rationale[id_]['rationale']

        openai_prompt = PROMPT_TEMPLATE.replace("$$QUESTION$$", question).replace("$$RESPONSE$$", prediction).replace("$$REFERENCE$$", rationale)

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
        rationale = id2rationale[id_]['rationale']

        try:
            final_results.append({
                "id": id_,
                "question": original_example['question'],
                "rationale": rationale,
                "prediction": original_example['prediction'],
                "output": example[1]["choices"][0]["message"]["content"]
            })
        except:
            skip_cnt += 1
            continue

    final_results = sorted(final_results, key=lambda x: int(x['id'].split('-')[1]))
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

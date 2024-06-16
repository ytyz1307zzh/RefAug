import pdb
import os
import json
import time
import shutil
import argparse

MCQ_TEMPLATE = """You will be given a multiple-choice question and a response from a student. You need to check which answer option is chosen by the student by reading his response. The student may mention the corresponding letter of the answer option, or directly write the content of the answer option. The corresponding letter can only be one of the following: A, B, C, D. Output the answer option that the student chooses no matter whether it is correct or not. If the student writes an answer that does not match any of the options, or if the student does not make a choice among the answer options, output "None".

## Example 1:
Question: The formula below is often used by project managers to compute $E$, the estimated time to complete a job, where $O$ is the shortest completion time, $P$ is the longest completion time, and $M$ is the most likely completion time.$$E=\\frac{O+4 M+P}{6}$$Which of the following correctly gives $P$ in terms of $E, O$, and $M$ ? \nAnswer Choices: (A) $P=6 E-O-4 M$ (B) $P=-6 E+O+4 M$ (C) $P=\\frac{O+4 M+E}{6}$ (D) $P=\\frac{O+4 M-E}{6}$
Student's Response: Step-by-step solution:\n\nWe have the equation $$E=\\frac{O+4M+P}{6}.$$ Multiplying both sides by 6 gives $$6E = O+4M+P.$$ Subtracting 4M from both sides gives $$6E - 4M = O + P.$$ Finally, adding O to both sides gives $$P = 6E - 4M + O.$$ The correct answer is $\\boxed{A}$. The answer is A.
Student's Choice: A

## Example 2:
Question: $$y=a(x-2)(x+4)$$In the quadratic equation above, $a$ is a nonzero constant. The graph of the equation in the $x y$-plane is a parabola with vertex $(c, d)$. Which of the following is equal to $d$ ? \nAnswer Choices: (A) $-9 a$ (B) $-8 a$ (C) $-5 a$ (D) $-2 a$
Student's Response: Step-by-step solution:\n\nThe vertex of a parabola given by $y=a(x-h)^2+k$ is $(h, k)$. Thus, the vertex of the parabola given by $y=a(x-2)(x+4)$ is $(2, k)$, where $k$ is the constant term when the expression is expanded. We find $k$ by multiplying $(x-2)(x+4)$ out: $(x-2)(x+4)=x^2-16$. Thus, $k=-16$. The answer is $\\boxed{-8a}$. The answer is -8a.
Student's Choice: B
Explanation: The student answers -8a, which matches with (B) $-8 a$, although the format is slightly changed.

## Example 3:
Question: $$x^{2}+20 x+y^{2}+16 y=-20$$The equation above defines a circle in the $x y$-plane. What are the coordinates of the center of the circle? \nAnswer Choices: (A) $(-20,-16)$ (B) $(-10,-8)$ (C) $(10,8)$ (D) $(20,16)$
Student's Response: Step-by-step solution:\n\nCompleting the square in both variables, we have $(x+10)^2 + (y-8)^2 = 256$. Thus, the center of the circle is $\\boxed{(-10, 8)}$. The answer is (-10,8)
Student's Choice: None
Explanation: The student answers (-10, 8). However, there is no such option in the answer choices.

## Now solve the following example. You don't need to write the explanation. Just output the corresponding letter or "None".
Question: $$QUESTION$$
Student's Response: $$RESPONSE$$
Student's Choice:"""


NUMERIC_TEMPLATE = """Please read the following examples, then extract the answer (arabic numerals) from the model response. If no answer is provided, write "None".

Question: A glass of milk is 8 ounces of milk.  John drinks 2 glasses of milk.  If milk has 3 calories per ounce how many calories did he consume?\n\nHe drank 2*8=16 ounces of milk\nThat means he drank 16*3=48 calories\nThe answer is 48.\n\nConsidering John now wants to maintain a balanced diet and needs to consume 300 calories from milk per day, how many glasses of milk does he need to drink daily?\n\nIf 8 ounces of milk has 3 calories, then 1 ounce of milk has 3/8 = 0.375 calories\nTo consume 300 calories, John needs to drink 300 / 0.375 = 800 ounces of milk\nSince 1 glass of milk is 8 ounces, John needs to drink 800 / 8 = 100 glasses of milk\nThe answer is 100.\n\nNow, if John wants to increase his daily calorie intake from milk by 20% to support his workout routine, how many more glasses of milk does he need to drink compared to the original 300-calorie plan?

Model response: Step-by-step solution:\n\nFirst, calculate the number of calories John needs to consume daily for the increased intake. Since a 20% increase means adding 20/100*300 = 60 calories to his daily intake, John now needs to consume 300+60 = 360 calories from milk per day.\n\nNext, determine how many ounces of milk John needs to drink to consume 360 calories. Since 1 ounce of milk has 0.375 calories, John needs 360 / 0.375 = 960 ounces of milk.\n\nFinally, since 1 glass of milk is 8 ounces, John needs to drink 960 / 8 = 120 glasses of milk per day.\n\nTherefore, John needs to drink 120 - 100 = 20 more glasses of milk compared to his original 300-calorie plan.

Extracted answer: 20

Question: If a classroom has 3 times as many girls as they do boys, and 1/10 as many nongendered children as they do boys, and the classroom has 30 boys.  How many total children does it have?\n\nThe classroom has 30*3=90 girls.\nThe classroom has 30/10=3 nongendered children.\nThe classroom has 30+90+3=123 children total.\nThe answer is 123.\n\nGiven that information, if each girl in the classroom has 2 pencils and each boy has 3 pencils, while the nongendered children have 1 pencil each, how many pencils are there in total in the classroom?\n\nThe classroom has 90 girls, who have 2 pencils each, so 90*2=180 pencils.\nThe classroom has 30 boys, who have 3 pencils each, so 30*3=90 pencils.\nThe classroom has 3 nongendered children, who have 1 pencil each, so 3*1=3 pencils.\nIn total, the classroom has 180+90+3=273 pencils.\nThe answer is 273.\n\nNow, suppose each child in the class needs to form groups for a project, with each group consisting of 3 girls, 2 boys, and 1 nongendered child. What is the maximum number of such groups that can be formed?

Model response: Step-by-step solution:\n\nFirst, calculate the total number of children in the classroom. We know there are $g$ girls, $b$ boys, and $n$ nongendered children. The total number of children is $g + b + n$.\n\nEach group consists of 3 girls, 2 boys, and 1 nongendered child. The number of groups that can be formed is determined by the combination of these children, which can be calculated using the combination formula $\\binom{n}{r} = \\frac{n!}{r!(n-r)!}$, where $n$ is the total number of items, $r$ is the number of items to choose, and $!$ denotes factorial.\n\nIn this case, $n = g + b + n$, $r = 3 + 2 + 1 = 6$, and we want to find the number of ways to choose 6 children out of $g + b + n$ to form a group.\n\nThe number of groups that can be formed is $\\binom{g + b + n}{6} = \\frac{(g + b + n)!}{6!(g + b + n - 6)!}$.\n\nThis formula gives us the maximum number of groups that can be formed, considering the constraints of the problem.

Extracted answer: None

Question: There are 36 penguins sunbathing in the snow.  One-third of them jump in and swim in the ocean.  Another one-third go inside the cave to eat their dinner.  How many penguins are still left sunbathing?\n\n36/3 = 12 penguins go in the ocean\n36/3 = 12 penguins go in the cave\n36 - 12 - 12 = 12 penguins are still sunbathing\nThe answer is 12.\n\nGiven that each penguin eaten by a penguin contributes to a 2% increase in their probability of catching a fish on their next swim, and every penguin starts with a base chance of 10% to catch a fish, what is the probability of a penguin catching a fish after it has eaten 3 times?\n\nFirst, we need to determine the probability of a penguin catching a fish after eating 1 time. We start with a base probability of 10% and multiply it by 110% (100% + 20%) to account for the increased chance of catching a fish, resulting in a probability of 12%.\nThen, we multiply this probability by itself three times to account for the three times the penguin eats, resulting in 12% * 12% * 12% = 2.16%.\nThe answer is 2.16%.\n\nIf the probability of catching a fish continues to increase at the same rate with each meal and a penguin decides to eat 15 times before swimming in the ocean, what is the new probability of catching a fish, assuming it does not exceed 100%?

Model response: Step-by-step solution:\n\nFirst, we calculate the probability after one meal as described in the original problem, which is a 2.16% increase from the base 10% chance.\nThen, we calculate the probability after two meals, which is 2.16% * 2.16% = 4.60% increase from the base 10% chance.\nAfter three meals, the probability is 2.16% * 4.60% = 9.77% increase from the base 10% chance.\nAfter four meals, the probability is 2.16% * 9.77% = 20.49% increase from the base 10% chance.\nAfter five meals, the probability is 2.16% * 20.49% = 42.15% increase from the base 10% chance.\nAfter six meals, the probability is 2.16% * 42.15% = 88.37% increase from the base 10% chance.\nAfter seven meals, the probability is 2.16% * 88.37% = 181.51% increase from the base 10% chance, which exceeds 100% and is not possible.\n\nTherefore, the new probability of catching a fish after 15 meals is 100%, the maximum possible probability.

Extracted answer: 100

Question: $$QUESTION$$

Model response: $$RESPONSE$$

Extracted answer:"""


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
    parser.add_argument("-data_path", default="test-greedy-output-extracted.json", help='The path to model predictions (after removing the reflective section)')
    parser.add_argument("-request_save_path", default="temp_save_request.jsonl")
    parser.add_argument("-response_save_path", default="temp_save_response.jsonl")
    parser.add_argument("-output_path", default="test-greedy-output-extracted-mmlu-sat.json")
    parser.add_argument("-datasets", nargs='+', help='List of datasets to process', required=True, help='If evaluating MMLU and SAT, pass "mmlu sat"')
    parser.add_argument("-data_type", type=str, choices=["mcq", "numeric"], default="mcq")
    parser.add_argument("-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-max_tokens", type=int, default=32)
    parser.add_argument("-temperature", type=float, default=0.0)
    parser.add_argument("-top_p", type=float, default=0.0)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-url", type=str, default="https://api.openai.com/v1/chat/completions", help='URL to OpenAI endpoint')
    parser.add_argument("-api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("-max_tpm", type=int, default=100000, help="Maximum tokens per minute")
    parser.add_argument("-max_rpm", type=int, default=500, help="Maximum requests per minute")
    parser.add_argument("-tokenizer", type=str, default="cl100k_base", help='tiktoken tokenizer')
    parser.add_argument("-max_attempts", type=int, default=10, help="Max number of retries if failed")
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

    all_requests = []
    for example in data:
        id_ = example['id']

        if id_.split("-")[0] not in args.datasets:
            continue

        question = example['instruction']
        response = example['output']

        id2example[id_] = {
            "question": question,
            "response": response
        }

        if args.data_type == "mcq":
            template = MCQ_TEMPLATE
        elif args.data_type == "numeric":
            template = NUMERIC_TEMPLATE

        openai_prompt = template.replace("$$QUESTION$$", question).replace("$$RESPONSE$$", response)
        messages = [
            {"role": "system", "content": "You are a helpful assistant and good at math."},
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
                "instruction": original_example['question'],
                "original_output": original_example['response'],
                "output": example[1]["choices"][0]["message"]["content"]
            })
        except:
            if len(example[1]) == args.max_attempts and \
                eval(example[1][-1])["error"]["code"] in ["context_length_exceeded", "invalid_prompt"]:

                print(f'Context length exceeded for instance {id_}, setting the predicted answer to "None"!')
                final_results.append({
                    "id": id_,
                    "instruction": original_example['question'],
                    "original_output": original_example['response'],
                    "output": "None"
                })
            else:
                print(f'[ERROR] Failed to process {id_}!')
                skip_cnt += 1
            
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

from vllm import LLM, SamplingParams
import torch

import argparse
import json
import sys
sys.path.append('src/data')
from prompt_template import ALTERNATIVE_FOLLOWUP_REFLECT_TEMPLATE


parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str, default="train-qa-format.json", help='A JSON file with all Q-A data. Questions should in a field called "question" and answers in a field called "response". Also need a "id" field to identify the instance.')
parser.add_argument('-output', type=str, default='train-reflection-llama3.json')
parser.add_argument('-model', type=str, help='Huggingface model path')
parser.add_argument('-temperature', type=float, default=0.7)
parser.add_argument('-top_p', type=float, default=1.0)
parser.add_argument('-max_tokens', type=int, default=2048)
args = parser.parse_args()

llm = LLM(
    model=args.model,
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
    dtype=torch.bfloat16
)
tokenizer = llm.get_tokenizer()

data = json.load(open(args.data, "r"))
print(f'Loaded {len(data)} data from {args.data}')

all_requests = []
for example in data:
    id_ = example['id']
    question = example['question']
    response = example['response']

    system_prompt = "You are a helpful assistant and good at math."
    user_prompt = ALTERNATIVE_FOLLOWUP_REFLECT_TEMPLATE.replace("$$QUESTION$$", question).replace("$$RESPONSE$$", response)

    messages = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        tokenize=False,
    )

    # Print one example out
    if len(all_requests) == 0:
        print('*' * 15 + '\n\n' + messages + '\n\n' + '*' * 15)

    all_requests.append(messages)

print(f'Collected {len(all_requests)} requests')

sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # Add EOT as termination token
)

outputs = llm.generate(
    all_requests,
    sampling_params
)

assert len(outputs) == len(data)
results = []
for j in range(len(outputs)):
    reflection = outputs[j].outputs[0].text.strip()
    example = data[j]

    example['reflection'] = reflection
    results.append(example)

json.dump(results, open(args.output, 'w'), ensure_ascii=False, indent=4)
print(f'Saved {len(results)} to {args.output}')

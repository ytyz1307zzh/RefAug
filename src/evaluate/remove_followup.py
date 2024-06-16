import json
import re
import pdb
import argparse
from tqdm import tqdm


def check_repeated_ending(text):
    text_length = len(text)
    
    # Start from 1-gram
    for n in range(1, text_length // 6):
        n_gram = text[-n:]
        # Count the number of times the n-gram is repeated
        count = 1
        for i in range(text_length - n, -1, -n):
            if text[i - n:i] == n_gram:
                count += 1
            else:
                break
        
        # Check if the n-gram is repeated more than 5 times
        if count > 5:
            return True, n_gram, count
    
    return False, '', 0


parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str, default='test-greedy-output.json')
parser.add_argument("-output", type=str, default='test-greedy-output-extracted.json')
args = parser.parse_args()

data = json.load(open(args.data, 'r', encoding='utf8'))
format_error = 0
box_cnt = 0
repeat_error = 0

error_results = []
for example in tqdm(data):
    output = example['output']
    error_flag = False

    if check_repeated_ending(output)[0]:
        repeat_error += 1

    # Extract the output by indicator of the start of reflective section
    if "Reflection:" in output:
        output = output.split("Reflection:")[0].strip(" \n#*")

    # Extract the output by indicator of the predicted answer
    elif 'The answer is ' in output or "the answer is " in output:
        try:
            matches = [(m.start(0), m.end(0)) for m in re.finditer(r'[Tt]he answer is (.*?)(\n|$)', output)]
            output = output[:matches[0][1]].strip(" \n#*")
        except:
            error_flag = True

    # Extract the output by indicator of the predicted answer (in boxed format like MATH)
    elif "\\boxed" in output:
        box_cnt += 1
        try:
            matches = [(m.start(0), m.end(0)) for m in re.finditer(r'(\\boxed{.*?})(\.|\n|$|\$)', output)]
            output = output[:matches[0][1]].strip(" \n.$")
        except:
            error_flag = True

    else:
        error_flag = True
        
    if error_flag:
        format_error += 1
        error_results.append({
            "id": example['id'],
            "output": output
        })

    example['output'] = output

json.dump(data, open(args.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
print(f"Saved {len(data)} examples to {args.output}")
print(f"Format Extraction Error: {format_error}")
print(f"Repeated Ending Error: {repeat_error}")
print(f"Boxed Count: {box_cnt}")

json.dump(error_results, open("error.json", 'w', encoding='utf8'), indent=4, ensure_ascii=False)
print(f"Saved {len(error_results)} error examples to error.json")

import re
import pdb
import math
import json
import argparse
from copy import deepcopy
from statistics import mean


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # remove parentheses
    string = string.replace("\\(", "")
    string = string.replace("\\)", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    # if string == "0.5":
    #    string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None


def number_it(num: str):
    if num.count("^") == 1:
        num = num.replace("^", "**")
    if 'frac' in num:
        pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
        num = re.sub(pattern, r"\1/\2", num)
        try:
            num = str(eval(num))
        except Exception:
            pass
    elif ',' in num:
        num = num.replace(',', '')

    if floatify(num) is not None:
        return floatify(num)
    else:
        try:
            num = eval(num)
            if isinstance(num, list) or isinstance(num, tuple):
                num = num[0]
            if floatify(num) is not None:
                return floatify(num)
            else:
                return None
        except Exception:
            return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            if gt == 0:
                return p == 0
            else:
                return round(p) == gt
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_both_string_and_number_format(answer, groundtruth_str, groundtruth_num):
    if answer == groundtruth_str:
        return True
    else:
        if groundtruth_num is not None and number_it(answer) is not None:
            if compare_two_numbers(number_it(answer), groundtruth_num):
                return True
            else:
                return False
        else:
            return False


def extract_math_answer(pred_str):
    if 'The answer is ' in pred_str :
        pred = pred_str.split('The answer is ')[-1].strip()
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if not ans:
            return ""
        if ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0: break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a

    else:
        pattern = r'-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if len(pred) >= 1:
            pred = pred[-1]
        else: pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
    pred=_strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if not ans:
            return ""
        if ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0: break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
    return pred


def delete_extra_zero(n):
    '''Delete extra zeros in the end of a decimal'''
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # Delete extra zeros in the end
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # Transform to int if possible
        n=str(n)
        return n


def answer_clean(dataset: str, answer_trigger: str, pred: str):

    if dataset == "mathematics":
        dataset = "deepmind"

    pred = pred.strip('\n')

    if dataset == "math":
        if len(pred) > 0:
            pred_final=extract_math_answer(pred)
            return pred_final
        else:
            return pred

    if answer_trigger not in pred and '\\boxed' in pred:
        
        ans = pred.split('\\boxed')[-1]
        if not ans:
            pred = ""
        else:
            if ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0: break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _strip_string(a)
            pred=a

        answer_flag = True if len(pred) > 1 else False
    
    else:
        # Split the trigger to find the answer.
        preds = re.split(answer_trigger, pred)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if '=' in pred:
        pred = pred.split('=')[-1].strip()

    if dataset in ("aqua", "sat") or "mmlu" in dataset:
        tmp = re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
        if tmp:
            pred = tmp
        else:
            pred = [pred.strip().strip('.')]
    elif dataset in ("gsm8k", "svamp", "simuleq"):
        pred = pred.replace(",", "")
        pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    elif dataset in ("numglue",):
        tmp = re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
        if tmp:
            pred = tmp
        else:
            pred = pred.replace(",", "")
            pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    elif dataset in ("mawps", "deepmind"):
        num_pred = deepcopy(pred[:-1]) if pred.endswith('.') else deepcopy(pred)
        num_pred = number_it(num_pred)
        
        if num_pred is not None:
            return num_pred
        else:
            pred = pred.replace(",", "")
            pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    return pred


def data_reader(dataset_dir, dataset: str):
    results = []
    decoder = json.JSONDecoder()

    if dataset == "mathematics":
        dataset = "deepmind"

    if dataset == "aqua":
        with open(f'{dataset_dir}/AQuA/AQuA.json') as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                question = json_res["question"].strip() + "\n" + choice
                answer = json_res["correct"]
                results.append({
                    "id": json_res["id"],
                    "question": question,
                    "answer": answer
                })
    elif dataset == 'math':
        with open(f'{dataset_dir}/math/MATH.json', 'r') as f:
            loaded = json.load(f)
        for d in loaded:
            question = d['question']
            answer = d['answer']
            results.append({
                "id": d["id"],
                "question": question,
                "answer": answer
            })
    elif dataset == "gsm8k":
        with open(f'{dataset_dir}/gsm8k/gsm8k.jsonl') as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                question = json_res["question"].strip()
                answer = delete_extra_zero(json_res["answer"].split("#### ")[-1].replace(",", ""))
                results.append({
                    "id": json_res["id"],
                    "question": question,
                    "answer": answer
                })
    elif dataset == "svamp":
        with open(f'{dataset_dir}/SVAMP/SVAMP.json') as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                question = q
                answer = delete_extra_zero(a)
                results.append({
                    "id": line["id"],
                    "question": question,
                    "answer": answer
                })
    elif 'mmlu' in dataset:
        with open(f'{dataset_dir}/mmlu/mathematics.json') as f:
            json_data = json.load(f)
            for line in json_data:
                options = f'(A) {line["choices"][0]} (B) {line["choices"][1]} (C) {line["choices"][2]} (D) {line["choices"][3]}'
                q = line["question"] + '\n' + 'Answer Choices: ' + options
                a = ['A', 'B', 'C', 'D'][line['answer']]
                question = q
                answer = a
                results.append({
                    "id": line["id"],
                    "question": question,
                    "answer": answer
                })
    elif dataset in ['numglue', 'simuleq', 'deepmind', 'sat']:
        with open(f'{dataset_dir}/{dataset}/{dataset}.json') as f:
            json_data = json.load(f)
            for line in json_data:
                assert isinstance(line['question'], str) and isinstance(line['question'], str), line
                question = line['question']
                answer = str(line['answer'])
                results.append({
                    "id": line["id"],
                    "question": question,
                    "answer": answer
                })
    elif dataset == "mawps":
        with open(f'{dataset_dir}/mawps/mawps.json') as f:
            json_data = json.load(f)
            for line in json_data:
                question = line["sQuestion"].strip()
                assert len(line["lSolutions"]) == 1
                answer = str(line["lSolutions"][0])
                if answer[-2:] == ".0":
                    answer = answer[:-2]
                results.append({
                    "id": line["id"],
                    "question": question,
                    "answer": answer
                })
    else:
        raise ValueError("dataset is not properly defined ...")

    # print(f"dataset : {dataset}")
    # print(f"data size : {len(results)}")

    return results


def eval_accuracy(dataset, ground_truth, pred_answer):
    if dataset == "mathematics":
        dataset = "deepmind"

    if dataset == 'math':
        assert len(ground_truth) == 2, ground_truth
        groundtruth_str, groundtruth_num = ground_truth
        if compare_both_string_and_number_format(pred_answer, groundtruth_str, groundtruth_num):
            accuracy = True
        else:
            accuracy = False
    elif dataset == "mawps":
        ground_truth = number_it(ground_truth)
        assert ground_truth is not None
        if isinstance(pred_answer, str):
            pred_answer = number_it(pred_answer)
        if pred_answer is not None and compare_two_numbers(pred_answer, ground_truth):
            accuracy = True
        else:
            accuracy = False
    elif dataset == "deepmind":
        ground_truth_num = number_it(ground_truth)
        # If ground-truth is a number, compare in the form of numbers
        if ground_truth_num is not None:
            if isinstance(pred_answer, str):
                pred_answer = number_it(pred_answer)
            if pred_answer is not None and compare_two_numbers(pred_answer, ground_truth_num):
                accuracy = True
            else:
                accuracy = False
        else:
            accuracy = (pred_answer == ground_truth)
    else:
        if pred_answer == ground_truth:
            accuracy = True
        else:
            accuracy = False

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='data/MathInstruct', help='Path to all MathInstruct test datasets')
    parser.add_argument('-pred', type=str, default='test-greedy-output.json', help='Path to the prediction results')
    parser.add_argument('-output', type=str, default=None, help='Path to the output file')
    args = parser.parse_args()

    all_predictions = json.load(open(args.pred, 'r', encoding='utf8'))
    dataset2pred = {}

    for example in all_predictions:
        id_ = example['id']
        dataset_name, id_ = id_.split('-')

        if dataset_name not in dataset2pred:
            dataset2pred[dataset_name] = []

        example['id'] = int(id_)
        dataset2pred[dataset_name].append(example)

    results = []
    for dataset_name in dataset2pred:
        dataset = data_reader(args.data_dir, dataset_name.lower())
        # print(f'Found {len(dataset)} examples in dataset {dataset_name}')
        
        dataset = sorted(dataset, key=lambda x: x['id'])
        predictions = sorted(dataset2pred[dataset_name], key=lambda x: x['id'])

        # For debugging only
        # pred_ids = [x['id'] for x in predictions]
        # dataset = [x for x in dataset if x['id'] in pred_ids]

        accuracy_list = []
        for data, pred in zip(dataset, predictions):
            assert data['id'] == pred['id'] and data['question'] == pred['instruction']
            output = pred['output']
            pred_answer = answer_clean(dataset_name.lower(), 'The answer is', output)
            ground_truth = data['answer']

            accuracy = eval_accuracy(dataset_name.lower(), ground_truth, pred_answer)

            results.append({
                'id': f"{dataset_name}-{data['id']}",
                'question': data['question'],
                'answer': ground_truth,
                'output': output,
                'pred_answer': pred_answer,
                'accuracy': accuracy
            })
            accuracy_list.append(accuracy)

        print(f'Dataset {dataset_name} ({len(accuracy_list)}) final accuracy: {mean(accuracy_list):.2%}', end=", ")
        print()
    
    if args.output:
        json.dump(results, open(args.output, 'w', encoding='utf8'), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

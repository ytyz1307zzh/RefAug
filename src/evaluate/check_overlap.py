import datasets
import argparse
from nltk.util import ngrams
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import nltk.util
import pdb


def load_test_dataset(data_path):
    id2example = {}

    data = json.load(open(data_path, "r", encoding='utf-8'))

    for example in data:
        dataset = example['id'].split('-')[0]
        if dataset not in ['MATH', 'gsm8k']:
            continue

        if dataset == "gsm8k":
            response = example['answer']
            # Remove all text wrapped with <<>> in response
            response = re.sub(r'<<.+?>>', '', response)
            response = re.sub(r'#### ', "The answer is ", response)

        if dataset == "MATH":
            response = example['solution']

        id2example[example['id']] = {
            'id': example['id'],
            'input': example['question'],
            'output': response,
            'meta': {
                "dataset": dataset
            }
        }

    return id2example


def make_test_index(id2example, key):
    # Compute:
    #   id -> {-1 : [full text string],
    #          30   : [30gram1_string, 30gram2_string, ...],
    #          20   : ...
    #          .. }
    index = defaultdict(lambda: defaultdict(list))
    for id_, obj in tqdm(id2example.items(), total=len(id2example)):
        text = obj[key]
        index[id_][-1].append(text)

        for n in [30, 20, 15, 10]:
            obj_ngrams = ngrams(text.split(), n)
            obj_ngrams = list(set([' '.join(ngram) for ngram in obj_ngrams]))
            index[id_][n] = obj_ngrams
    return index


def save_index(id2example, index, output_dir, key, subset=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if subset is None:
        subset = 'gsm_MATH'

    with open(os.path.join(output_dir, f'{subset}_{key}_test_index.json'), 'w') as f:
        json.dump({
            'id2example': id2example,
            'index': index
        }, f, indent=4)


def save_hits(hits_ds, output_dir, field, test_key, ngram_n, subset=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if subset is None:
        subset = 'gsm_MATH'

    outfile = os.path.join(
        output_dir,
        f'{subset}_%s_%s_hits_%d.json' % (field, test_key, ngram_n)
    )
    hits_ds.to_json(outfile)


def _check_full_string(text, test_ngrams, test_id, ngram_n):
    hits = []
    for ngram in test_ngrams:
        if ngram in text:
            hits.append({
                'test_id': test_id,
                'ngram_n': ngram_n,
                'ngram': ngram
            })
    return hits


def _check_intersection(text_ngrams_set, test_ngrams, test_id, ngram_n):
    hits = []
    intersection = text_ngrams_set.intersection(set(test_ngrams))
    for ngram in intersection:
        hits.append({
            'test_id': test_id,
            'ngram_n': ngram_n,
            'ngram': ngram
        })
    return hits


def get_hits(example, index, column, ngram_n):
    # Map an example to a list of hits. A hit means that some n-gram
    # (with n = ngram_n, or a full string when ngram_n=-1) from the test
    # set occurs in the text found in example[column]. The hit records
    # the matched text and the id of the test example.
    text = example[column]
    # HACK: Add a placeholder, otherwise `datasets` crashes with empty `hits`
    hits = [{
        'test_id': "None",
        'ngram_n': -123,
        "ngram": "[PLACEHOLDER]"
    }]

    if ngram_n != -1:
        text_ngrams = ngrams(text.split(), ngram_n)
        text_ngrams_set = set([' '.join(ngram) for ngram in text_ngrams])

    for test_id in index:
        n2ngrams = index[test_id]
        test_ngrams = n2ngrams[ngram_n]
        if ngram_n == -1:
            hits_ = _check_full_string(text, test_ngrams, test_id, ngram_n)
        else:
            hits_ = _check_intersection(text_ngrams_set, test_ngrams, test_id, ngram_n)
        hits.extend(hits_)
    return {
        "train_id": example['id'],
        'hits': hits
    }


def flatten(data, id2example):
    out = {}
    for k, v in data.items():
        if k != 'hits':
            out[k] = v
    for hit in data['hits']:
        if hit['test_id'] != "None":
            for k, v in hit.items():
                out[k] = v
            for k, v in id2example[hit['test_id']].items():
                out[k] = v
    return out


def main(args):
    # Load test dataset and make an index of n-grams
    test_ds = load_test_dataset(args.test_path)

    if args.subset == "gsm8k":
        test_ds = {k: v for k, v in test_ds.items() if v['meta']['dataset'] == 'gsm8k'}
    elif args.subset == "MATH":
        test_ds = {k: v for k, v in test_ds.items() if v['meta']['dataset'] == 'MATH'}

    print(f'Loaded {len(test_ds)} examples from {args.test_path}')

    test_index = make_test_index(test_ds, key=args.test_key)
    save_index(test_ds, test_index, args.output_dir, args.test_key, args.subset)

    # Load dataset and find n-gram matches
    ds = datasets.load_dataset("json", data_files={"train": args.data_path})['train']

    if args.subset == "gsm8k":
        ds = ds.filter(
            lambda x: x['id'].startswith('GSM'),
            batched=False,
        )
    elif args.subset == "MATH":
        ds = ds.filter(
            lambda x: x['id'].startswith('MATH'),
            batched=False,
        )

    print(f'Loaded {len(ds)} examples from {args.data_path}')

    for ngram_n in args.ngram_n:
        print("##### ngram_n %d" % ngram_n)
        hits_ds = ds.map(
            lambda x: get_hits(x, test_index, args.field, ngram_n),
            batched=False,
            num_proc=args.n_processes
        )

        hits_ds = hits_ds.filter(
            lambda x: len(x['hits']) > 1,
            batched=False,
            num_proc=args.n_processes
        )
        print("%d hits" % len(hits_ds))

        hits_ds = hits_ds.map(
            lambda x: flatten(x, test_ds),
            batched=False,
            num_proc=args.n_processes
        )
        save_hits(hits_ds, args.output_dir, args.field, args.test_key, ngram_n, args.subset)


if __name__ == '__main__':
    # If one wants to check overlap in questions, select "input" as test-key, gsm8k or MATH training file as data-path, and question as field.
    # If one wants to check overlap in responses, select "output" as test-key, gsm8k or MATH training file as data-path, and response as field.
    # If one wants to check overlap between reflection and responses, select "output" as test-key, the RefAug data file as data-path, and reflection (or whatever you used to store the reflective section in your data file) as field.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-path', type=str, default='data/original/test.json'
    )
    parser.add_argument(
        '--test-key', type=str, default='input', choices=['input', 'output'],
        help='Specifies whether to check overlap with test inputs or test outputs.'
    )
    parser.add_argument('--subset', type=str, default=None, choices=['gsm8k', 'MATH'])
    parser.add_argument('--data-path', type=str, default='train-reflection.json')
    parser.add_argument('--field', type=str, default='response', choices=['question', 'response', 'reflection'],
    help='Which field do you want to check contamination?')
    parser.add_argument('--n-processes', type=int, default=16)
    parser.add_argument(
        '--ngram-n', type=int, nargs='+',
        default=30,
        choices=[10, 20, 30, -1],
        help='Ngram overlap(s) to check. -1 for full match of `--test-key`. Usually 20 for question and 30 for response.'
    )
    parser.add_argument('--output-dir', type=str, default='overlap_analysis', help='Output directory')

    args = parser.parse_args()
    main(args)
import argparse
import json
import os

def merge_lists(input_files):
    merged_list = []
    for file_path in input_files:
        with open(file_path, "r", encoding='utf8') as f:
            data = json.load(f)
            print(f"Read {len(data)} data from {file_path}")
            merged_list.extend(data)

    return merged_list

def main():
    parser = argparse.ArgumentParser(description='Merge lists from multiple JSON files into one JSON file.')
    parser.add_argument('-input_prefix', type=str, required=True,
                        help='Paths to the input JSON files')
    parser.add_argument('-n', type=int, required=True,
                        help='Number of parts to split the list into')
    parser.add_argument('-output', type=str, required=True,
                        help='Path to the output JSON file')
    args = parser.parse_args()

    input_files = [f"{args.input_prefix}_part_{i}.json" for i in range(0, args.n)]

    merged_list = merge_lists(input_files)
    merged_list = sorted(merged_list, key=lambda x: x['id'])

    with open(args.output, "w", encoding='utf8') as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(merged_list)} examples to {args.output}")
    

if __name__ == "__main__":
    main()

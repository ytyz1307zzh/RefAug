import json
import random
import argparse

def split_list(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help='Path to the JSON file')
    parser.add_argument("-n", type=int, help="Number of parts to split the list into")
    parser.add_argument("-output_dir", type=str, help="Output directory")
    parser.add_argument("-output_prefix", type=str, help="Output file prefix", default="test")
    args = parser.parse_args()

    data = json.load(open(args.data, "r", encoding='utf8'))
    print(f"Read {len(data)} data from {args.data}")
    random.shuffle(data)

    split_data = list(split_list(data, args.n))

    for i, part in enumerate(split_data):
        json.dump(part, open(f"{args.output_dir}/{args.output_prefix}_part_{i}.json", "w", encoding='utf8'), indent=4, ensure_ascii=False)
        print(f"Saved {len(part)} examples to {args.output_dir}/{args.output_prefix}_part_{i}.json")


if __name__ == "__main__":
    main()

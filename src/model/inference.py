import argparse
import json
import logging
import os
import pdb
import random
import multiprocessing
from functools import partial
from datetime import datetime

import torch
import vllm
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

logger = logging.getLogger(__name__)
torch.set_printoptions(threshold=3000, sci_mode=False)


TORCH_PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VLLM_PRECISION_MAP = {
    "fp32": 'float32',
    "fp16": 'float16',
    "bf16": 'bfloat16',
}


def get_instruction(example):
    if 'instruction' in example:
        return example['instruction']
    elif 'question' in example:
        return example['question']
    elif "prompt" in example:
        return example["prompt"]
    elif "messages" in example:
        assert example['messages'][-1]['role'] == "user"
        return example["messages"][-1]["content"]
    else:
        raise ValueError("Either `instruction`, `question`, `prompt`, or `messages` should be in the instance.")


def concat_prompt_completion_format(example, splitter="\n\n"):
    formatted_text = example['prompt'].strip() + splitter

    if "response_prefix" in example:
        formatted_text += example["response_prefix"]
        
    return formatted_text


def concat_messages_format(example, tokenizer):

    formatted_text = ""
    for message in example['messages']:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))

    formatted_text += "<|assistant|>\n"

    if "response_prefix" in example:
        formatted_text += example["response_prefix"]

    return formatted_text


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1536,
        help="Max sequence length for the instruction.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for sampling-based generation",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for sampling, 1.0 means no penalty",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help='If specified, only use the first "num_examples" examples in the dataset.',
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="If specified, overwrite the original output file (if exists).",
    )
    parser.add_argument(
        "--continue_output",
        action="store_true",
        help="If specified, continue writing to the original output file (if exists).",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput.")
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    if args.seed is not None:
        set_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )

    logger.info("loading data and model...")
    # load some data
    eval_data = json.load(open(args.data, "r", encoding="utf-8"))

    if args.num_examples is not None:
        eval_data = eval_data[: args.num_examples]

    logger.info(f"Total evaluation data: {len(eval_data)}")

    prev_data = None
    if os.path.exists(args.output_path) and not args.overwrite_output:
        if args.continue_output:
            prev_data = json.load(open(args.output_path, "r", encoding="utf-8"))
            prev_data_ids = {x["id"] for x in prev_data}
            logger.warning(
                f"Continue writing to {args.output_path}, which already has {len(prev_data)} examples..."
            )
            eval_data = [x for x in eval_data if x["id"] not in prev_data_ids]
            if len(eval_data) == 0:
                logger.warning("No remaining examples to generate, exiting...")
                return
        else:
            logger.warning("File %s already exists, exiting...", args.output_path)
            return

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Handle the input format
    if "prompt" in eval_data[0]:

        splitter = "\n\n"
        input_process_function = partial(
            concat_prompt_completion_format,
            splitter=splitter,
        )

    elif "messages" in eval_data[0]:
        input_process_function = partial(
            concat_messages_format,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError("Either `messages` or `prompt` should be in the instance.")

    # vLLM decoding
    if args.use_vllm:
        logger.info(f"Using vLLM decoding with {torch.cuda.device_count()} GPUs!")

        model = vllm.LLM(
            model=args.model,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=VLLM_PRECISION_MAP[args.precision],
        )
        model.set_tokenizer(tokenizer)

        sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        
        input_texts = [input_process_function(example) for example in eval_data]

        logger.info(input_texts[0])

        vllm_generations = model.generate(input_texts, sampling_params)

        all_outputs = []
        for j in range(len(eval_data)):
            all_outputs.append({
                "id": eval_data[j]["id"],
                "instruction": get_instruction(eval_data[j]),
                "generator": f"{args.model}",
                "output": vllm_generations[j].outputs[0].text.strip(),
            })

        if prev_data is not None:
            all_outputs += prev_data
        
        all_outputs = sorted(all_outputs, key=lambda x: x["id"])
        json.dump(
            all_outputs,
            open(args.output_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )
        logger.info(f"Saved {len(all_outputs)} examples to {args.output_path}.")

        logger.info(all_outputs[0])

    # Accelerate decoding
    else:
        logger.info("Using Accelerate decoding!")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=TORCH_PRECISION_MAP[args.precision],
            trust_remote_code=args.trust_remote_code,
        )

        logger.info("model and data loaded!")
        logger.info("generating...")

        if args.temperature == 0.0:
            # Greedy decoding
            generation_config = GenerationConfig.from_pretrained(
                args.model,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            # sampling
            generation_config = GenerationConfig.from_pretrained(
                args.model,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=True,
                repetition_penalty=args.repetition_penalty,
            )
        logger.warning(
            f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Start generating..."
        )
        
        eval_data = [
            {
                "id": example['id'],
                "input": input_process_function(example),
            }
            for example in eval_data
        ]
        random.shuffle(eval_data)
        my_outputs = []  # Generations for the current process

        with accelerator.split_between_processes(eval_data) as eval_data_curr_process:

            dataloader = torch.utils.data.DataLoader(
                eval_data_curr_process, batch_size=args.batch_size, shuffle=False
            )

            with torch.inference_mode():
                for samples in tqdm(dataloader, desc=f"GPU {accelerator.process_index}"):

                    # All inputs must be pre-formatted (special tokens, system prompt, etc)
                    input_texts = samples['input']

                    inputs = tokenizer(
                        input_texts,
                        return_tensors="pt",
                        max_length=args.max_input_length,
                        padding=True,
                        truncation=True,
                    )
                    input_ids = inputs.input_ids.to(model.device)
                    attention_mask = inputs.attention_mask.to(model.device)

                    # print the first example
                    if len(my_outputs) == 0:
                        logger.info(input_ids[0])

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                    )

                    for j in range(len(samples["id"])):
                        output = outputs[j]
                        output_string = tokenizer.decode(
                            output[input_ids.size(1) :], skip_special_tokens=True
                        )
                        my_outputs.append(
                            {
                                "id": samples["id"][j].item(),
                                # TODO: this will lead to an error if we need to extract the "messages" field from samples
                                "instruction": get_instruction(samples)[j],
                                "generator": f"{args.model}",
                                "output": output_string.strip(),
                            }
                        )

            output_path_curr_process = args.output_path + f".{accelerator.process_index}"
            json.dump(
                my_outputs,
                open(output_path_curr_process, "w", encoding="utf8"),
                indent=4,
                ensure_ascii=False,
            )

        logger.warning(
            f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Finished generation!"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # concatenate outputs from all processes
            all_outputs = []
            for i in range(accelerator.num_processes):
                output_path_curr_process = args.output_path + f".{i}"
                all_outputs += json.load(
                    open(output_path_curr_process, "r", encoding="utf-8")
                )
                os.remove(output_path_curr_process)

            if prev_data is not None:
                all_outputs += prev_data

            all_outputs = sorted(all_outputs, key=lambda x: x["id"])
            json.dump(
                all_outputs,
                open(args.output_path, "w", encoding="utf8"),
                indent=4,
                ensure_ascii=False,
            )
            print(f"Saved {len(all_outputs)} examples to {args.output_path}.")

            logger.info(all_outputs[0])
            # format should be something like:
            # {'instruction': 'What are the names of some famous actors that started their careers on Broadway?', 'input': '', 'output': 'Some famous actors that started their careers on Broadway are Hugh Jackman, Meryl Streep, Denzel Washington, Audra McDonald, and Lin-Manuel Miranda.', 'generator': 'gpt-3.5-turbo-0301', 'dataset': 'helpful_base', 'datasplit': 'eval'}


if __name__ == "__main__":
    main()
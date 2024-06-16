import argparse
import logging
import math
import os
import pdb
import random
from datetime import datetime
from functools import partial

import datasets
import deepspeed
import huggingface_hub
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    get_scheduler,
)

logger = get_logger(__name__)
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
torch.set_printoptions(threshold=3000, sci_mode=False)

VLLM_PRECISION_MAP = {
    "fp32": 'float32',
    "fp16": 'float16',
    "bf16": 'bfloat16',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--do_valid",
        action="store_true",
        help="If passed, will run validation on the validation dataset.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--save_merged_lora_model",
        action="store_true",
        help="If passed, will merge the lora modules and save the entire model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_valid_batch_size",
        type=int,
        default=None,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--peak_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--final_learning_rate",
        type=float,
        default=0,
        help="Final learning rate when the training ends. This controls how the lr will decay.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If passed, gradient checkpointing will be turned on for lower memory footprint.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--save_at_epoch",
        type=str,
        default=None,
        help='Specify which epochs to save, separated by commas. For example, "1,2,3" will save at the end of the first, second and third epochs. Only effective when checkpointing_steps=epoch.',
    )
    parser.add_argument(
        "--save_best_on_valid",
        action="store_true",
        help="If passed, will save the best model on the validation dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--wandb_username",
        type=str,
        default=None,
        help="The username for wandb.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
        "--hf_token_file",
        type=str,
        default=None,
        help="The file with the huggingface login token if the model to load has access limitations.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json",
                "jsonl",
            ], "`train_file` should be a json/jsonl file."
    return args


def save_model(
    accelerator,
    tokenizer,
    model,
    output_dir: str,
    use_lora: bool,
):
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to mannually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, splitter="\n\n"):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space

    example_text = example['prompt'].strip() + splitter + example['completion'].strip()
    example_text = example_text + tokenizer.eos_token

    tokenized_example = tokenizer(
        example_text,
        return_tensors='pt',
        max_length=max_seq_length,
        truncation=True
    )

    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    prompt_prefix = example['prompt'].strip() + splitter
    tokenized_prompt = tokenizer(
        prompt_prefix,
        return_tensors='pt',
        max_length=max_seq_length,
        truncation=True
    )

    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """

    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):

        if message["role"] != "assistant":
            
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]

            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])

            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]

            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main():
    args = parse_args()

    if args.hf_token_file is not None:
        with open(args.hf_token_file, "r", encoding="utf8") as fin:
            hf_token = fin.read().strip()
        huggingface_hub.login(hf_token)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # add logging to 'train.log'
    logger.logger.addHandler(
        logging.FileHandler(os.path.join(args.output_dir, "train.log"), "w")
    )
    logger.log(INFO, accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # print args and save it to a file
    logger.log(INFO, args)
    if accelerator.is_local_main_process:
        with open(os.path.join(args.output_dir, "args"), "w", encoding="utf8") as fout:
            print(args, file=fout)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.valid_file is not None:
            data_files["validation"] = args.valid_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        )
    else:
        logger.log(INFO, "Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if args.gradient_checkpointing:
        logger.log(
            INFO,
            f"Model supports gradient checkpointing: {model.supports_gradient_checkpointing}",
        )
        model.gradient_checkpointing_enable()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embedding_size:
            logger.log(WARN, f"Resizing token embeddings from {embedding_size} to {len(tokenizer)}!")
            model.resize_token_embeddings(len(tokenizer))
        embedding_size = model.get_input_embeddings().weight.shape[0]  # Update embedding_size, probably used in sum reduction of loss

    if args.use_lora:
        logger.log(INFO, "Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        logger.log(INFO, "Encoding data with prompt-completion format!")
        
        splitter = "\n\n"

        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            splitter=splitter,
        )
    elif "messages" in raw_datasets["train"].column_names:
        logger.log(INFO, "Encoding data with chat messages format!")
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

    with accelerator.main_process_first():
        assert (
            raw_datasets["train"].column_names
            == raw_datasets["validation"].column_names
        )
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )

        oversize_cnt = len(
            [
                ex
                for ex in lm_datasets["train"]
                if len(ex["input_ids"]) >= args.max_seq_length
            ]
        )
        logger.log(INFO, f'Total examples: {len(lm_datasets["train"])}')
        logger.log(INFO, f"Oversize examples: {oversize_cnt}")

        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(
            lambda example: (example["labels"] != -100).any()
        )

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.log(INFO, f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8
        ),
        batch_size=args.per_device_train_batch_size,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8
        ),
        batch_size=args.per_device_valid_batch_size \
            if args.per_device_valid_batch_size is not None \
                else args.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.peak_learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )
    num_warmup_steps = round(num_training_steps_for_scheduler * args.warmup_ratio)

    # If we want to let the final learning rate (at the end of decay) to be higher than 0, we need to let the scheduler "think" we have more steps
    if args.final_learning_rate > 0:
        w = args.warmup_ratio * 100
        k = args.peak_learning_rate / args.final_learning_rate
        n = num_training_steps_for_scheduler
        num_training_steps_for_scheduler = round((100 * k - w) * n / ((k - 1) * 100))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=num_warmup_steps
    )

    # Prepare everything with `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )  # global steps
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs (only useful when specifying --max_train_steps instead of --num_train_epochs)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers(
            project_name="celt",
            config=experiment_config,
            init_kwargs={
                "wandb": {"entity": args.wandb_username, "name": "/".join(args.output_dir.split("/")[-2:])}
            },
        )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.log(INFO, "***** Running training *****")
    logger.log(INFO, f"  Num examples = {len(train_dataset)}")
    logger.log(INFO, f"  Num Epochs = {args.num_train_epochs}")
    logger.log(
        INFO,
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}",
    )
    logger.log(
        INFO,
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}",
    )
    logger.log(
        INFO, f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    logger.log(INFO, f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    if args.checkpointing_steps == "epoch" and args.save_at_epoch is not None:
        epochs_to_save = [int(x) for x in args.save_at_epoch.strip().split(",")]
        logger.log(INFO, f"Saving at the end of epochs: {sorted(epochs_to_save)}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(args.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    best_valid_loss = float("inf")
    past_logging_steps = 0  # how many steps since last logging

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0

        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
               
                if args.reduce_loss == 'mean':
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                past_logging_steps += 1

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / past_logging_steps
                    )
                    logger.log(
                        INFO,
                        f"[{datetime.now().strftime('%H:%M:%S')}]"
                        f"  Global step: {completed_steps}, Local step: {step}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}",
                    )
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    past_logging_steps = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        save_flag = True  # whether to save the model

                        # Evaluate the model on validation set
                        if args.do_valid:
                            accelerator.wait_for_everyone()
                            valid_loss = evaluate(
                                args, valid_dataloader, model, accelerator
                            )
                            if valid_loss < best_valid_loss:
                                logger.log(
                                    INFO,
                                    f"New best validation loss: {best_valid_loss} --> {valid_loss}!",
                                )
                                best_valid_loss = valid_loss
                            else:  # this is not the best checkpoint
                                logger.log(
                                    INFO,
                                    f"Current validation loss: {valid_loss}, best validation loss: {best_valid_loss}",
                                )
                                if args.save_best_on_valid:
                                    save_flag = False

                            if args.with_tracking:
                                accelerator.log(
                                    {"valid_loss": valid_loss.item()},
                                    step=completed_steps,
                                )

                        if save_flag:
                            output_dir = os.path.join(
                                args.output_dir, f"step_{completed_steps}"
                            )
                            if args.do_valid and args.save_best_on_valid:
                                output_dir = os.path.join(args.output_dir, "best_ckpt")
                            accelerator.wait_for_everyone()
                            save_model(
                                accelerator,
                                tokenizer,
                                model,
                                output_dir=output_dir,
                                use_lora=args.use_lora,
                            )
                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":

            if args.save_at_epoch is None or str(
                epoch + 1
            ) in args.save_at_epoch.strip().split(","):
                save_flag = True  # whether to save the model
            else:
                save_flag = False

            # Evaluate the model on validation set
            if args.do_valid:
                accelerator.wait_for_everyone()
                valid_loss = evaluate(args, valid_dataloader, model, accelerator)
                if valid_loss < best_valid_loss:
                    logger.log(
                        INFO,
                        f"New best validation loss: {best_valid_loss} --> {valid_loss}!",
                    )
                    best_valid_loss = valid_loss
                else:  # this is not the best checkpoint
                    logger.log(
                        INFO,
                        f"Current validation loss: {valid_loss}, best validation loss: {best_valid_loss}",
                    )
                    if args.save_best_on_valid:
                        save_flag = False

                if args.with_tracking:
                    accelerator.log(
                        {"valid_loss": valid_loss.item()}, step=completed_steps
                    )

            if save_flag:
                output_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
                if args.do_valid and args.save_best_on_valid:
                    output_dir = os.path.join(args.output_dir, "best_ckpt")
                accelerator.wait_for_everyone()
                save_model(
                    accelerator,
                    tokenizer,
                    model,
                    output_dir=output_dir,
                    use_lora=args.use_lora,
                )

    if args.with_tracking:
        accelerator.end_training()

    # Save the final model
    accelerator.wait_for_everyone()
    # If we save the model after every epoch, then the "final model" has already been saved
    if args.checkpointing_steps != "epoch":
        save_model(
            accelerator,
            tokenizer,
            model,
            output_dir=args.output_dir,
            use_lora=args.use_lora,
        )


def evaluate(args, valid_dataloader, model, accelerator):
    model.eval()
    total_loss = 0.0
    total_data_cnt = 0
    logger.log(INFO, "Start evaluating on the validation data...")

    for _, batch in enumerate(valid_dataloader):
        with torch.inference_mode():
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # We keep track of the loss at each logged step
            data_cnt = len(batch["input_ids"])
            total_loss += loss.detach().float() * data_cnt
            total_data_cnt += torch.tensor(data_cnt).to(accelerator.device)

    # Gather loss and data_cnt from all devices
    accelerator.wait_for_everyone()
    total_loss = accelerator.gather(total_loss).sum()
    total_data_cnt = accelerator.gather(total_data_cnt).sum()
    valid_loss = total_loss / total_data_cnt
    logger.log(INFO, "Evaluation finished!")

    model.train()
    return valid_loss


if __name__ == "__main__":
    main()
# Copyright (c) Facebook, Inc. and its affiliates.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import numpy as np
import json
from tqdm import tqdm

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action="store_true", help="avoid using CUDA when available")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--model_name_or_path", type=str, default="output_model_gpt2_100epoch", help="path to pre-trained model or shortcut name")
parser.add_argument("--input", type=str, help="input text file, each line corresponding to one instance")
parser.add_argument("--output", type=str, help="output file")
parser.add_argument("--eos_token_id", type=int, default=None, help="eos token id")

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

set_seed(args)

ckpt = args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(ckpt, padding_side='left', use_safetensors=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True)
model.to(args.device)

context_token = tokenizer.encode('<|context|>', return_tensors='pt')
endofcontext_token = tokenizer.encode(' <|endofcontext|>', return_tensors='pt')

with open(args.input, "r") as f:
    prompts = f.read().strip().split("\n")

print(prompts)
batch_size = 1
ret = []

for batch in range(len(prompts)):
    prompt_text = prompts[batch]
    print("Prompt:::", prompt_text)

    prompt_text_tokenized = tokenizer.encode(prompt_text, return_tensors='pt')

    model_input = prompt_text_tokenized
    attention_mask = torch.ones_like(model_input)

    out_tokenized = model.generate(model_input, max_length=1024, eos_token_id=50258, pad_token_id=50260, attention_mask=attention_mask).tolist()[0]
    out_str = tokenizer.decode(out_tokenized)
    out_str = out_str.split('\n')[0]

    print('Out str:::', out_str)
    print()
    ret.append(out_str)

with open(args.output, "w") as f:
    json.dump(ret, f, indent=2)

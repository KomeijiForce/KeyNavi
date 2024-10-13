from transformers import AutoTokenizer, AutoModelForCausalLM
from kmeans_pytorch import kmeans
import torch
from sklearn.metrics import f1_score
from collections import Counter

import json
import re
import numpy as np
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
generator = AutoModelForCausalLM.from_pretrained(model_path)
generator.eval()
generator = generator.to("cuda")
generator.eval()

def probe(seq):
    input_ids = tokenizer(seq, return_tensors="pt").input_ids.to("cuda")
    for _ in range(256):
        logits = generator(input_ids).logits[0, -1]
        idx = logits.argsort(descending=True)[0]
        next_token = tokenizer.decode(idx)
        input_ids = torch.cat((input_ids, idx.unsqueeze(0).unsqueeze(0)), -1)
        seq += next_token
        if "#" in next_token:
            break

    if len(re.findall(r"### Way 1:\n(.*)?###", seq, flags=re.DOTALL)) > 0:
        return re.findall(r"### Way 1:\n(.*)?###", seq, flags=re.DOTALL)[-1]
    else:
        return "nothing"

def decode(for_decode):
    seq = ""
    for _ in range(8):
        logits = generator(**tokenizer(for_decode, return_tensors="pt").to("cuda")).logits[0, -1]
        idx = logits.argsort(descending=True)[0]
        next_token = tokenizer.decode(idx)
        if "\"" in next_token:
            break
        seq += next_token
        for_decode += next_token
        
    seq = "".join(re.findall(r'\d+', seq))

    return seq

def verbalize_escape(escape_list):
    return "".join(f"### Way {idx+1}:\n{escape}\n" for idx, escape in enumerate(escape_list)) + f"### Way {1+len(escape_list)}:\n* **Step 1:**\n"

from datasets import load_dataset

accs = []
accs_path = []

my_answers = []

dataset = [data for data in load_dataset("openai/gsm8k", "main")["test"]]

cnt = 0
n_cot = 4
topk=100

bar = tqdm(dataset)

for data in bar:
    
    cnt += 1

    question = data["question"]
    answer = data["answer"].split("#### ")[-1]

    instruction = f"{question}\n\nPlease show me some different ways to solve this problem."
    escape_list = []
    probed = []
    probed_ids = []
    all_decoded = []
    
    with torch.no_grad():
        for run in range(n_cot):
        
            prompt = f'''{instruction}

Here are different ways to solve the problem:
{verbalize_escape([])}'''

            logits = generator(**tokenizer(prompt, return_tensors="pt").to("cuda"), output_hidden_states=True).logits[0, -1]
            ids = logits.argsort(descending=True)[:topk]
            
            for idx in ids:
                next_token = tokenizer.decode(idx)
                if next_token not in probed:
                    probed.append(next_token)
                    prompt = f'''{instruction}

Here are different ways to solve the problem:
{verbalize_escape([])}'''
                    entity = probe(prompt+next_token)
                    probed_ids.append(idx)
                    escape_list.append(entity)
                    for_decode = f'''{instruction}

Here are different ways to solve the problem:
''' + entity + "**Final Answer (Only Number):** \""
                    all_decoded.append(decode(for_decode))

                    print(all_decoded)
                    
                    break

    my_answer = Counter(all_decoded).most_common(1)[0][0]
    my_answers.append(all_decoded)
    acc = int(my_answer == answer)
    acc_path = np.mean([ans==answer for ans in all_decoded])
    accs.append(acc)
    accs_path.append(acc_path)
    bar.set_description(f"#NO.{cnt} #Acc={np.mean(accs)*100:.4} #Acc Path={np.mean(accs_path)*100:.4}")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import f1_score

import json
import re
import numpy as np
from tqdm import tqdm
from time import sleep
from copy import deepcopy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

icn = True
n_text = 100

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
generator = AutoModelForCausalLM.from_pretrained(model_path)
generator.eval()
generator = generator.to("cuda")

probe_cache = {}

def probe(seq):
    if seq in probe_cache:
        return probe_cache[seq]
        
    for _ in range(64):
        logits = generator(**tokenizer(seq, return_tensors="pt").to("cuda")).logits[0, -1]
        probs = logits.softmax(-1)
        idx = logits.argsort(descending=True)[0]
        next_token = tokenizer.decode(idx)
        seq += next_token
        if "\"" in next_token:
            break

    if len(re.findall("\"(.*)?\"", seq)) > 0:
        ans = re.findall("\"(.*)?\"", seq)[-1]
    else:
        ans = "nothing"
        
    probe_cache[seq] = ans

    return ans

def score(answer, instruction):

    if answer in cache:
        return cache[answer]

    response = openai.ChatCompletion.create(
    model=model_engine,
    temperature=0.0,
    messages=[{"role": "user", "content": f'''{instruction}\n\nIs "{answer}" considered as among the correct answers? Answer only "Yes" or "No".'''}],
    ).choices[0]['message']["content"]
    sleep(2.0)
    if "yes" in response.lower():
        cache[answer] = 1
        return 1
    else:
        cache[answer] = 0
        return 0

def verbalize_escape(escape_list):
    return "".join(f"{idx+1}. \"{escape}\"\n" for idx, escape in enumerate(escape_list)) + f"{1+len(escape_list)}. \""

def mapk(res, k=10):
    
    return np.mean([np.mean(res[:i+1]) for i in range(k)])

def random_probe(seq):
    if seq in probe_cache:
        return probe_cache[seq]
        
    for _ in range(64):
        logits = generator(**tokenizer(seq, return_tensors="pt").to("cuda")).logits[0, -1]
        probs = logits.softmax(-1)
        idx = torch.multinomial(probs, 1)[0]
        next_token = tokenizer.decode(idx)
        seq += next_token
        if "\"" in next_token:
            break

    if len(re.findall("\"(.*)?\"", seq)) > 0:
        ans = re.findall("\"(.*)?\"", seq)[-1]
    else:
        ans = "nothing"
        
    probe_cache[seq] = ans

    return ans

instruction = "Please write some positive sentences."
escape_list = []
n_tol = 10
n_run = 10
reg=True

escape_list = deepcopy(escape_list)

probed = [tokenizer.tokenize(escape)[0] for escape in escape_list]
golds = []

init_prompt = f'''{instruction}

Answer:
{verbalize_escape(escape_list)}'''

with torch.no_grad():
    bar = tqdm(range(n_run))
    for run in bar:

        tol = n_tol

        prompt = f'''{instruction}

Answer:
{verbalize_escape(escape_list if icn else [])}'''

        items = generator(**tokenizer(prompt, return_tensors="pt").to("cuda"), output_hidden_states=True)
        initial_state = items.hidden_states[-1][0, -1]
        lm_head_weight = generator.lm_head.weight
        # lm_head_weight = torch.nn.functional.normalize(lm_head_weight)
        logits = (initial_state.unsqueeze(0) * lm_head_weight).sum(-1)
        probs = logits.softmax(-1)
        ids = logits.argsort(descending=True)
        next_tokens = [tokenizer.decode(idx) for idx in ids]
        X = generator.lm_head.weight[ids]
        
        for x, next_token in zip(X, next_tokens):
            if next_token not in probed:
                probed.append(next_token)
                if reg:
                    entity = probe(init_prompt+next_token)
                else:
                    entity = probe(prompt+next_token)
                print(entity)
                escape_list.append(entity)
                tol -= 1
                if tol == 0:
                    break

json.dump(escape_list, open("positive.zerogen.icn.json", "w"))

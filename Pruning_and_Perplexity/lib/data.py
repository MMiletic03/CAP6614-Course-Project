# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import os
import glob
from datasets import load_dataset

# ----------------ORIGINAL----------------------#
# Description: Sets the random seed for both numpy
# and PyTorch to ensure reproducible calibration sample
# selection across runs.
#------------------------------------------------#
# Inputs: seed (int)
# Outputs: N/A
#------------------------------------------------#
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# ----------------ORIGINAL----------------------#
# Description: Thin wrapper that gives a raw token
# tensor the same .input_ids interface as a HuggingFace
# tokenizer output. Used so that eval_ppl_wikitext can
# access testenc.input_ids regradless of whether testenc
# came directly from a tokenizer or from a manual tensor
#------------------------------------------------#
# Inputs: input_ids (torch.Tensor)
# Outputs: TokenizerWrapper instance with .input_ids attribute
#------------------------------------------------#
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# ----------------MODIFIED----------------------#
# Description: Loads the wikitext2 dataset from local
# parquet files, tokenizes, the full train and test splits,
# and returns them in the format expected by the pruning and
# evalutation pipeline.
#------------------------------------------------#
# Inputs: nsamples (int), seed (int), seqlen (int), tokenizer
# Outputs: (trainloader (list of nsamples tuples of (input_ids,),
# each shape (1, seqlen)), testenc (torch.tensor of shape (1,N)))
#------------------------------------------------#
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    print("WT2(local): entered get_wikitext2()", flush=True)

    # ========== CHANGE TO REFLECT YOUR LOCAL DIRECTORY CONTAINING .parquet FILES ========== #
    local_dir = "/workspace/wanda/lib/wikitext-2-raw-v1"
    assert os.path.isdir(local_dir), f"Missing local dir: {local_dir}"

    # Load all three splits from local parquet files
    train_files = [os.path.join(local_dir, "train-00000-of-00001.parquet")]
    valid_files = [os.path.join(local_dir, "validation-00000-of-00001.parquet")]
    test_files  = [os.path.join(local_dir, "test-00000-of-00001.parquet")]

    ds = load_dataset("parquet", data_files={"train": train_files,"validation": valid_files,"test": test_files})
    print("WT2(local): rows:", ds["train"].num_rows, ds["test"].num_rows, flush=True)

    # Concatenate all articles with "\n\n" seperator and tokenize as one string
    trainenc = tokenizer("\n\n".join(ds["train"]["text"]), return_tensors="pt")
    testenc  = tokenizer("\n\n".join(ds["test"]["text"]),  return_tensors="pt")
    print(f"WT2(local): train tokens={trainenc.input_ids.shape[1]}, test tokens={testenc.input_ids.shape[1]}", flush=True)

    # Build trainloader as random chunks for calibration
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        # Pick a random starting position that leaves room for a full seqlen window
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i + seqlen]

        # Including tar for compatibility with training-style data pipelines
        tar = inp.clone()
        tar[:, :-1] = -100

        # Wrap in a tuple so batch[0] returns the full (1, seqlen) tensor.
        trainloader.append((inp,))

    # Return the full test token stream as a raw (1, N) tensor.
    print("WT2: done", flush=True)
    return trainloader, testenc.input_ids

# ----------------MODIFIED----------------------#
# Description: Streams the C4 English training split
# and collects nsamples documents that tokenize to at
# least seqlen tokens. Each document is truncated to 
# exactly seqlen tokens.
#------------------------------------------------#
# Inputs: nsamples (int), seed (int), seqlen (int), tokenizer
# Outputs: (trainloader, None)
#------------------------------------------------#
def get_c4(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    # Streaming=True yields one document at a time without downloading the corpus
    dataset = load_dataset("allenai/c4","en",split="train",streaming=True)
    trainloader = []
    for i, data in enumerate(dataset):
        if len(trainloader) == nsamples:
            break
            
        text = data["text"]

        # Skip very short documents
        if len(text) < 10:
            continue
        enc = tokenizer(text,return_tensors="pt",max_length=seqlen,truncation=True,).input_ids

        # Skip documents that tokenize to fewer than seqlen tokens
        if enc.shape[1] < seqlen:
            continue
        trainloader.append((enc[:, :seqlen],))
        
    return trainloader, None

# ---------------- ORIGINAL----------------------#
# Description: Dispatcher that routes to the correct
# dataset loader based on the dataset name string.
# Called by both the pruning functions and the evalutation function
#------------------------------------------------#
# Inputs: name (str), nsamples (int), seed (int), seqlen (int), tokenizer
# Outputs: (trainloader, testloader) 
#------------------------------------------------#
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import pprint
import random
import time
from pprint import pprint
import cutlass # NVIDIA Cutlass for GEMM testing

''' 
This file uses NVIDIA Cutlass for GEMM operations to test inference speed on Ampere-and-newer series NVIDIA GPUs 
to test the difference in structured pruned linear matrix multlication vs unpruned operations.
'''

''' This function comes from the Wanda repo - main.py '''
def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
    ).cuda().eval()

    model.seqlen = model.config.max_position_embeddings 
    return model


''' This function came from the Wanda repo - prune.py'''
def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

''' 
This function is based on:
https://ipd.graylab.jhu.edu/rfdiffusion2/cutlass-3.5.1/python/docs/externals/00_basic_gemm.html
'''
def benchmark_GEMM(layer_dict, batch_size, seqlen, is_sparse, warmups=10, dtype=np.float16):
    keys = list(layer_dict.keys())
    times = []
    # Test every layer in the dictionary
    for i in range(len(keys)):
        key = keys[i]
        layer = layer_dict[key] 
        print(f"Starting benchmark for {key}")
        print_module = False # tells GEMM not to print all the printouts
        infeats = layer.in_features
        outfeats = layer.out_features

        # Define random matrices of the right shapes
        M, K, N = batch_size * seqlen, infeats, outfeats

        A = torch.randn(M, K, dtype=torch.float16).cuda()
        B = torch.tensor(layer.weight.T.detach().cpu().float().numpy(), dtype=torch.float16).cuda() # weight matrix (transposed)
        C = torch.zeros(M, N, dtype=torch.float16).cuda()
        D = torch.zeros(M, N, dtype=torch.float16).cuda()

        # This controls whether the C++ GEMM declaration will be printed at each step. Set to `false` to omit this information.
        print_module = False

        # Create a GEMM plan
        if is_sparse == True:
            plan = cutlass.SparseGemm(element=dtype, layout=cutlass.LayoutType.RowMajor)
        else:
            plan = cutlass.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor)

        # Warmup the GEMM (ensure all initialization is complete, no DRAM fetches, etc)
        for _ in range(10):
            plan.run(A, B, C, D, print_module=print_module)

        
        # Now test it for real
        torch.cuda.synchronize() # block CPU operations so timing does not get messed up due to memory movement
        start_time = time.perf_counter() # ns
        plan.run(A, B, C, D, print_module=print_module)
        torch.cuda.synchronize()
        end_time = time.perf_counter() # ns
        time_ms = (end_time - start_time) * 1000; # ms
        times.append(time_ms)
    
    return times 

# Load in the model. For testing purposes, we can stream it from huggingface.
testing = True
if testing == True:
    print("Loading model from hf")
    # login to huggingface
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    login(token=HF_TOKEN)
    # load llm from hf
    model_name = "meta-llama/llama-2-7b-hf"
    print(f"loading {model_name} from huggingface for testing.")
    cache_dir = "llm_weights"
    model = get_llm(model_name, cache_dir)
    model.eval()
    model_args = {
        "pretrained": model_name,
        "cache_dir": cache_dir,
        "tokenizer": model_name, # will auto-load the right tokenizer
        "max_length": 4096
    }

else:
    # Load in a pre-pruned model w/ tokenizer
    print("loading in pre-pruned model")
    ''' This code comes from the main.py file of Wanda. '''
    cache_dir = '/path/here' # no spaces
    tokenizer_dir = '/path/here' # no spaces
    print(f"loading local model for evaluation")

    model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True,
        ).cuda().eval()
    model.seqlen = model.config.max_position_embeddings 
    model.eval()
    model_args = {
        "pretrained": cache_dir,
        "tokenizer": tokenizer_dir,
        "max_length": 4096
    }
    
print("Model loaded!")

# Extract just the layers we care about computing according to the paper
# print(model)
all_layers = find_layers(model)
# q_proj layers
q_proj_layers = {key: val for key, val in all_layers.items() if 'q_proj' in key}
q_proj_layers = dict(sorted(q_proj_layers.items())) # sort by keys so layers are in order
# k_proj layers
k_proj_layers = {key: val for key, val in all_layers.items() if 'k_proj' in key}
k_proj_layers = dict(sorted(k_proj_layers.items()))
# v_proj layers
v_proj_layers = {key: val for key, val in all_layers.items() if 'v_proj' in key}
v_proj_layers = dict(sorted(v_proj_layers.items()))
# o_proj layers
o_proj_layers = {key: val for key, val in all_layers.items() if 'o_proj' in key}
o_proj_layers = dict(sorted(o_proj_layers.items())) 
# up_proj layers
up_proj_layers = {key: val for key, val in all_layers.items() if 'up_proj' in key}
up_proj_layers = dict(sorted(up_proj_layers.items())) 
# gate_proj layers
gate_proj_layers = {key: val for key, val in all_layers.items() if 'gate_proj' in key}
gate_proj_layers = dict(sorted(gate_proj_layers.items())) 
# down_proj layers
down_proj_layers = {key: val for key, val in all_layers.items() if 'down_proj' in key}
down_proj_layers = dict(sorted(down_proj_layers.items())) 

# Execute a GEMM test on each layer
# The paper combines q/k/v/o_proj, up/gate_proj, down_proj since they are the same size
batch_size = 1
seqlen = model.seqlen
if testing == True:
    print("Dense Model Inference Speed Results")
else:
    print("Pruned Sparse 2:4 Model Inference Speed Results")

q_times = benchmark_GEMM(q_proj_layers, batch_size, seqlen, is_sparse=False)
k_times = benchmark_GEMM(k_proj_layers, batch_size, seqlen, is_sparse=False)
v_times = benchmark_GEMM(v_proj_layers, batch_size, seqlen, is_sparse=False)
o_times = benchmark_GEMM(o_proj_layers, batch_size, seqlen, is_sparse=False)
times_self_att = [*q_times, *k_times, *v_times, *o_times]
ave_sa = sum(times_self_att) / len(times_self_att)


up_times = benchmark_GEMM(up_proj_layers, batch_size, seqlen, is_sparse=False)
gate_times = benchmark_GEMM(gate_proj_layers, batch_size, seqlen, is_sparse=False)
mlp_times = [*up_times, *gate_times]
ave_mlp = sum(mlp_times) / len(mlp_times)


down_times = benchmark_GEMM(down_proj_layers, batch_size, seqlen, is_sparse=False)
ave_down = sum(down_times) / len(down_times)
print(f"q/k/v/o_proj (ms): {ave_sa:.2f}")
print(f"up/gate_proj (ms): {ave_mlp:.2f}")
print(f"down_proj (ms): {ave_down:.2f}")






import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import fnmatch
from huggingface_hub import login
import os
import pprint
from lm_eval import evaluator 
from lm_eval.tasks import TaskManager

''' INSTALLS ''' 
# Run this file on the latest installs of the above packages! This is needed to reconcile dependency issues.
# pip install -r requirements_for_zero_shot.txt

''' This function was modified from the Wanda repo. - eval.py'''
task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
# task_list = ["winogrande", "openbookqa"] # short version for testing
def eval_zero_shot(model_name, model_args, task_list=task_list, 
        num_fewshot=0):
    
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    tm = TaskManager()
    all_tasks = tm.all_tasks
    task_names = pattern_match(task_list, all_tasks)
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    print("Starting zero shot evaluation")
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        use_cache=None,
        limit=limit,
        check_integrity=False,
        log_samples=False # do not save individual results, just aggregate
    )

    return results["results"] 

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

def main():
    '''We will need to get the pruned model weights and tokenizer from the pruning process.'''
    # use model.save_pretrained(save_dir) and tokenizer.save_pretrained(save_dir)
    
    # For testing, load in the model/tokenizer directly from huggingface
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
    
    # Evaluate the model on zero-shot characteristics
    results = eval_zero_shot(model_name, model_args)
    print("********************************")
    print("zero shot evaluation results")
    pprint(results)
    
    # Evaluate the model on few-shot dataset
    results = eval_zero_shot(model_name, model_args, task_list=["mmlu"], num_fewshot=5)
    print("few shot evaluation results")
    pprint(results)

if __name__ == "__main__":
    main()
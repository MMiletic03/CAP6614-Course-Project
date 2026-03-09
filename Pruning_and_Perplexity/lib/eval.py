# Import necessary modules
import time
import torch
import torch.nn as nn
from .data import get_loaders 
from collections import defaultdict
from lm_eval import tasks, evaluator
import fnmatch

# ---------------- MODIFIED----------------------#
# Description: Entry point for perplexity evaluation.
# Loads the wikitext2 test set as a single concatenated
# token tensor and evaluates the model using a sliding
# window over the entire test set. Matching the LLM perplexity
# evaluation methodology used in the Wanda paper.
#------------------------------------------------#
# Inputs: args, model, tokenizer, device
# Outputs: float (perplexity on wikitext2 test set)
#------------------------------------------------#
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    dataset = "wikitext2"
    print(f"evaluating on {dataset}")

    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
    )
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test

# ---------------- MODIFIED----------------------#
# Description: Evaluates perplexity over a list of
# fixed-lenth token chuncks. Computes cross-entropy
# loss over each chunk using teacher-forced next-token
# prediction, accumulates total NLL and token count,
# then returns exp(total_nll/total_tokens).
#------------------------------------------------#
# Inputs: model, trainloader (list of (input_ids,) tuples),
#         bs (batch size), device
# Outputs: float (perplexity)
#------------------------------------------------#
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    print("in eval_ppl_wikitext_train")
    nsamples = len(trainloader)
    print(f"nsamples {nsamples}")
    model.eval()
    # reduction="sum" gives total NLL across all tokens in the chunk,
    # allowing correct averaging over variable numbers of valid samples
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(nsamples):
            inputs = trainloader[i][0].to(device)  # (1, seqlen)
            out = model(inputs)
            logits = out.logits  # (1, seqlen, vocab)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()
            nll = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if not torch.isfinite(nll):
                # skip bad samples instead of breaking, so perplexity is computed over all valid samples
                print(f"[WARN] Non-finite NLL at sample {i}, nll={nll}. Skipping.", flush=True)
                continue
            total_nll += float(nll.item())
            total_tokens += int(shift_labels.numel())
            if i % 50 == 0:
                print(f"sample {i}", flush=True)
                
    if total_tokens == 0:
        # All samples produced non-finite NLL
        print("No tokens were evaluated (total_tokens=0). Returning inf perplexity.", flush=True)
        return float("inf")
    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)))
    print("wikitext perplexity", ppl, flush=True)
    torch.cuda.empty_cache()
    return ppl

# ---------------- ORIGINAL----------------------#
# Description: Evaluates perplexity by sliding a window
# of size model.seqlen over the entire concatenated test
# set, one non-overlapping chunk at a time.
#------------------------------------------------#
# Inputs: model, testenc (torch.Tensor of shape (1, N))
#         bs (batch size), device
# Outputs: float (perplexity)
#------------------------------------------------#
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    print("in eval_ppl_wikitext")

    # Print shape, dtype, or weird token values
    print(f"testenc shape: {testenc.shape}")
    print(f"testenc dtype: {testenc.dtype}")
    print(f"testenc min: {testenc.min()}, max: {testenc.max()}")
    nsamples = testenc.numel() // model.seqlen
    print(f"nsamples: {nsamples}, model.seqlen: {model.seqlen}")
    
    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# -----------------ORIGINAL----------------------#
# Description: Runs zero-shot evaluation on a set of 
# downstream tasks. Used to measure task performance
# degradation after pruning, complementing the perplexity
# metric. METHOD IS UNUSED.
#------------------------------------------------#
# Inputs: pretrained_path (str), task_list (list of str)
#         num_fewshot (int), use_accelerate (bool)
# Outputs: dict of lm_eval results per task
#------------------------------------------------#
def eval_zero_shot(
    pretrained_path,
    task_list=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"],
    num_fewshot=0,
    use_accelerate=True,
):
    print("in eval_zero_shot (lm-eval 0.4.2, loading from)", pretrained_path)

    import fnmatch
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_manager = TaskManager()
    task_names = task_list

    model_args = f"pretrained={pretrained_path}"
    if use_accelerate:
        model_args = f"pretrained={pretrained_path},use_accelerate=True"
    print("lm-eval tasks =", task_names, flush=True)
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size="auto",
        device="cuda",
        task_manager=task_manager,
        check_integrity=False,
        log_samples=False,
    )
    return results
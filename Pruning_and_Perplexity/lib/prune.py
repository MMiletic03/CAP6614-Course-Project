import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .ablate import AblateGPT

# ----------------MODIFIED----------------------#
# Description: Returns True only for Linear layers inside the transformer blocks
# that correspond to the main projection layers (q, k, v, o, gate, up, down).
# Excludes the LM head, embedding, and any non-standard layers 
# to prevent accidentally pruning layers that would damage the model.
#------------------------------------------------#
# Inputs: name (str), module (nn.Module)
# Outputs: True/False (bool)
#------------------------------------------------#
def _is_prunable_linear(name: str, module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    forbidden = ("lm_head", "classifier", "score", "embed_tokens", "wte", "tok_embeddings")
    if any(k in name for k in forbidden):
        return False
    if ".layers." not in name:
        return False
    allowed = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if not any(k in name for k in allowed):
        return False

    return True

# ----------------MODIFIED----------------------#
# Description: Utility function that prints the faction 
# of zero weights, shape, dtype, and device for a given tensor. 
#------------------------------------------------#
# Inputs: tag (str), name (str), W (torch.tensor)
# Outputs: N/A
#------------------------------------------------#
def _print_weight_sparsity(tag: str, name: str, W: torch.Tensor):
    with torch.no_grad():
        z = (W == 0).float().mean().item()
        print(f"{tag} {name}: zeros={z:.4f} shape={tuple(W.shape)} dtype={W.dtype} device={W.device}", flush=True)

# ----------------ORIGINAL----------------------#
# Description: Recursively traverses a module and collects
# all submodules whose type matches one of the types in
# 'layers'. Used to identify which Linear layers within
# each transformer block should be pruned. 
#------------------------------------------------#
# Inputs: module (nn.Module), layers (list of layer types), name (str) 
# Outputs: dict mapping layer name (str) -> layer module (nn.Module)
#------------------------------------------------#
def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

# ----------------ORIGINAL----------------------#
# Description: Iterates over all transformer layers,
# counts the total number of zero-valued weights vs.
# total weights, and prints per-layer sparsity.
# Used as a sanity check after pruning.
#------------------------------------------------#
# Inputs: model (nn.Module)
# Outputs: float (global sparsity ratio across all prunable layers)
#------------------------------------------------#
def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    #Global counters across the entire model
    count = 0 
    total_params = 0
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # per-layer counters for the per-layer sparsity print
        sub_count = 0
        sub_params = 0
        
        for name in subset:
            W = subset[name].weight.data
            # Count exact zeros (pruned weights are set to exactly 0.0)
            count += (W==0).sum().item()
            total_params += W.numel()
            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

# ----------------MODIFIED----------------------#
# Description: Runs a forward pass through the model up to (not including)
# the first transformer layer using a Catcher hook. This captures the hidden
# states that will be fed into layer 0 during calibration. These
# captured inputs are then used layer-by-layer during the pruning loop
# to collect activation statistics without re-running the full model each time. 
#------------------------------------------------#
# Inputs: model (nn.Module), dataloader (list of (input_ids,) tuples), device
# Outputs: inps: (nsamples, seqlen, hidden_size), 
#          outs: zeros tensor of same shape, used as output buffer during pruning.
#          attention_mask: (1, seqlen) or None
#          position_ids: (1, seqlen) or None
#------------------------------------------------#
def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # For multi-GPU models, use the device that holds the embedding layer,
    # since that is where the initial token IDs will be processed
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map is not None and "model.embed_tokens" in hf_map:
        device = hf_map["model.embed_tokens"]
        
    dtype = next(iter(model.parameters())).dtype

    # Determine nsamples from dataloader length; default to 128 if not iterable
    nsamples = 0
    try:
        nsamples = len(dataloader)
    except TypeError:
        nsamples = 128

    # Pre-allocate the input buffer. Each row will be filled by Catcher hook
    # with the hidden states produced by the embedding + positional encoding layers
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False

    # cache holds the attention_mask and position_ids extracted from the first
    # forward pass, along with a counter tracking how many samples have been captured
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        # Replaces layers[0] temporarily; captures the input embeddings and
        # attention metadata passed ot the first transformer layer, the raises
        # ValueError to abort the forward pass early (avoid wasting compute)
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # Use .get() to safely handle models that don't pass these kwargs
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    # Verify that the dataloader is returning correctly shaped token IDs
    first_batch = next(iter(dataloader))
    print(f"[diag] len(batch): {len(first_batch)}", flush=True)
    print(f"[diag] batch[0].shape: {first_batch[0].shape}", flush=True)
    print(f"[diag] batch[0].dtype: {first_batch[0].dtype}", flush=True)
    print(f"[diag] batch[0]: {first_batch[0]}", flush=True)

    # Install the Catcher in place of the first transformer layer
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass  # expected: Catcher raises a ValueError after capturing the input
        except Exception as e:
            # An unexpected exception (ex. OOM, device mismatch, etc.)
            print(f"ERROR during calibration forward pass: {type(e).__name__}: {e}", flush=True)
            layers[0] = layers[0].module  # restore before raising
            raise

    # Restore the original first layer now that all samples have been captured
    layers[0] = layers[0].module

    # Verify that the Catcher fired for every sample
    print(f"[prepare_calibration_input] captured {cache['i']}/{nsamples} samples", flush=True)
    print(f"[prepare_calibration_input] inps: min={inps.min():.4f} max={inps.max():.4f} mean={inps.mean():.4f}", flush=True)

    # outs is an empty buffer of the same shape as inps.
    # It will be filled layer-by-layer during the pruning loop as each layer processes inps
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    # Fallback if the model didn't return mask/position_ids
    if attention_mask is None:
        attention_mask = torch.ones((1, model.seqlen), dtype=torch.long, device=device)
    if position_ids is None:
        position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0)
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids

# ----------------ORIGINAL----------------------#
# Description: Used by the Wanda variant (--use_variant flag).
# Instead of pruning a fixed fraction of weights per row, this
# function finds a mask based on a cumulative sum threshold 
# alpha * row_sum. THe variant searches for the alpha that
# achieves the target sparsity via binary search in prune_wanda.
# Not used in standard Wanda pruning.
#------------------------------------------------#
# Inputs: alpha (float), sort_res (tuple form torch.sort), W_metric (tensor)
#         tmp_metric (cumulative sum tensor), sum_before(row sums tensor)
# Outputs: (W_mask, cur_sparsity) - boolean mask and achieved sparsity float
#------------------------------------------------#
def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    # For each row, compute the threshold = alpha * total_row_sum
    # Any weight whose cumulative sorted score is <= threshold gets masked
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    # Gather the actual score value at the mask boundary for each row
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            # Importance metric (|W|); no activation input used
            W_metric = torch.abs(W)
            if prune_n != 0:
                # N:M structured pruning: slide a window of size prune_m along
                # the input dimension; within each window, keep only the top
                # (prune_m - prune_n) weights and zero the bottom prune_n
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        # topk with largest=False returns the prune_n smallest indices
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # Unstructured: find the global score threshold such that exactly sparsity_ratio fraction of 
                # all weights fall below it. Some rows may be pruned more than others.
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

# ----------------MODIFIED----------------------#
# Description: Wanda pruning. For each linear layer, 
# computes the importance score as S_ij = |W_ij| * ||X_j||_2, 
#  where ||X_j||_2 is the RMS norm of the j-th input
# feature across calibration samples. Weights with 
# the lowest cores are zeroed out.
#------------------------------------------------#
# Inputs: args (with sparsity_ratio, nsamples, seed, use_varient),
#         model, tokenizer, device, prune_n (int), prune_m (int)
# Outputs: N/A
#------------------------------------------------#
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False
    model.eval()

    print("loading calibdation data")
    # Load 128 random C4 samples for calibration
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    # Capture layer-0 input embeddings for all calibration samples in one pass.
    # After this call, inps[j] holds the (seqlen, hidden_size) input to layer 0
    # for calibration sample j, and attention_mask/position_ids are ready to use.
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        print("inps stats:", inps.min(), inps.max(), inps.mean())
        # If inps is all zeros, the Catcher never fired
        print("cache filled:", (inps != 0).any())

    layers = model.model.layers
    hf_map = getattr(model, "hf_device_map", None)

    # Accumulates metric computation time across all layers
    total_metric_time = 0.0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        # Only prune nn.Linear layers
        subset = {n: m for n, m in subset.items() if isinstance(m, nn.Linear)}

        # For multi-GPU models, move the activation buffers 
        # to the device hosting the transformer layer
        if hf_map is not None and f"model.layers.{i}" in hf_map:
            dev = hf_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        # Create one WrappedGPT per linear layer.
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            # Closure capturing 'name' so each hook writes to the correct wrapper
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register one forward hook per linear layer in this transformer block
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Calibration forward pass: run all nsamples through this transformer layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]

        # Remove all hooks now that statistics have been collected
        for h in handles:
            h.remove()

        # Start timing metric computation and masking ONLY
        torch.cuda.synchronize()
        t_layer_start = time.perf_counter()

        for name in subset:
            print(f"pruning layer {i} name {name}", flush=True)

            W = subset[name].weight.data
            scaler_row = wrapped_layers[name].scaler_row

            # Check shape: scaler_row must have one entry per input feature
            if scaler_row.numel() != W.shape[1]:
                raise RuntimeError(
                    f"scaler_row size mismatch for layer {i} module {name}: "
                    f"scaler_row={scaler_row.numel()} vs in_features={W.shape[1]}"
                )

            if not torch.isfinite(scaler_row).all():
                # If NaN/Inf in scaler_row. Fallback to magnitude pruning
                n_bad = (~torch.isfinite(scaler_row)).sum().item()
                print(f"[WARN] {n_bad} non-finite values in scaler_row layer {i} {name}. Falling back to magnitude.", flush=True)
                W_metric = torch.abs(W).float()
            else:
                # Core Wanda importance score
                W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1))

            # Initialize mask to all False
            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                # Structured N:M pruning
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Unstructured per-row pruning
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            s_before = (W == 0).float().mean().item()
            # Zero out the pruned weights in-place. The weight tensor is modified
            # directly (not a copy), so the model's parameters are permanently updated
            subset[name].weight.data[W_mask] = 0
            s_after = (subset[name].weight.data == 0).float().mean().item()
            print(f"[sparsity] layer {i} {name}: before={s_before:.4f} after={s_after:.4f}", flush=True)

        # Synchronize before stopping timer to ensure all masking GPU ops are done
        torch.cuda.synchronize()
        t_layer_end = time.perf_counter()
        # End the timing

        layer_metric_time = t_layer_end - t_layer_start
        total_metric_time += layer_metric_time
        print(f"[timing] layer {i} metric computation: {layer_metric_time*1000:.2f} ms", flush=True)
        
        # Forward pass after pruning to update activations for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]

        # Swap inps and outs buffers. The outputs of the pruned layer become
        # the inputs for the next layer in the next loop iteration
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

# ----------------MODIFIED----------------------#
# Description: SparseGPT pruning. Uses second-order Hessian information
# to determine which weights to prune and how to update remaining weights
# to compensate. For each layer, collect input activations, compute the 
# inverse Hessian via Cholesky decomposition, then prune and update
# weights block-by-block (blockize=128).
#------------------------------------------------#
# Inputs: args (with sparsity_ratio, nsamples, seed), 
#         model, tokenizer, dev, prune_n (int), prune_m (int)
# Outputs: N/A
#------------------------------------------------#
@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # For multi-GPU models, capture inputs on the device hosting the embeddings
    if getattr(model, "hf_device_map", None) is not None and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # Pre-allocate input buffer: (nsamples, seqlen, hidden_size)
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        # Same Catcher pattern as perpare_calibration_input
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
            
    # Install Catcher, run all calibration samples through the model to fill inps,
    # then restore the original layer
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass # eexpected: Catcher raises ValueError after storing input
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    
    # Accumulates fasterprune time across all layers
    total_metric_time = 0.0
    
    for i in range(len(layers)):
        layer = layers[i]

        # Move activation buffers to the correct device for multi-GPU models
        if getattr(model, "hf_device_map", None) is not None and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        # Wrap each linear layer with SparseGPT
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Calibration forward pass: collect Hessian statistics
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # Begin timing
        torch.cuda.synchronize()
        t_layer_start = time.perf_counter()

        # Start SparseGPT pruning
        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        # End timing
        torch.cuda.synchronize()
        t_layer_end = time.perf_counter()

        layer_metric_time = t_layer_end - t_layer_start
        total_metric_time += layer_metric_time
        print(f"[timing] layer {i} fasterprune: {layer_metric_time*1000:.2f} ms", flush=True)

        # Forward pass with pruned weights to propagate updated activations
        # to the next layer (same layer-chaining logic as prune_wanda)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# -----------------MODIFIED----------------------#
# Description: Ablation study pruning function. Tests combinations of pruning metrics
# with weight update strategies. Used to isolate the contribution of each component of
# Wanda and SparseGPT.
#------------------------------------------------#
# Inputs: args (with sparsity_ratio, nsamples, seed), 
#         model, tokenizer, dev, prune_n (int), prune_m (int)
# Outputs: N/A
#------------------------------------------------#
@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # For multi-GPU models, use the device hosting the embedding layer
    if getattr(model, "hf_device_map", None) is not None and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        # Same early-abort Catcher as prune_sparsegpt
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
            
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]

        # Move buffers to the correct device for multi-GPU models
        if getattr(model, "hf_device_map", None) is not None and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        # Wrap each layer with AblateGPT
        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register hooks and run calibration forward pass to collect statistics
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            # Determine the pruning mask based on the ablation variant
            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            # fasterprune applies the mask (or computes it iteratively) and
            # optionally updates remaining weights to compensate for pruning error
            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        # Propagate this layer's pruned outputs as inputs to the next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
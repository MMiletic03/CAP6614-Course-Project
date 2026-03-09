import torch
import torch.nn as nn

# ----------------MODIFIED----------------------#
# Description: Wraps a single nn.Linear layer to 
# accumulate input activation statistic during the 
# calibration forward pass. Registered as a forward
# hook on each Linear layer in a transformer block via
# add_batch(), which is called once per calibration sample.
# After all calibration samples have been processed, scaler_row
# holds E[x_j^2] for each input feature j, which is used to compute
# the Wanda pruning score in prune_wanda.
#------------------------------------------------#
class WrappedGPT:

    # ----------------MODIFIED----------------------#
    # Description: Initializes the wrapper for a given
    # Linear layer. scaler_row is allocated on the same 
    # device as the layer weights and sized to match the 
    # number of input features (columns of the weight matrix)
    # nsamples tracks the total number of token vectors seen so
    # far, used to correctly weight each new batch in the running
    # mean update.
    #------------------------------------------------#
    # Inputs: name (str), module (nn.Module)
    # Outputs: True/False (bool)
    #------------------------------------------------#
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    # ----------------MODIFIED----------------------#
    # Description: Called once per calibration sample via
    # a forward hook. Flattens the input to (n_tokens, in_features),
    # computes sum of squared activations per input feature, then updates scaler_row
    # using a running mean fomrula.
    #------------------------------------------------#
    # Inputs: inp (torch.Tensor), out (torch.Tensor)
    # Outputs: N/A
    #------------------------------------------------#
    def add_batch(self, inp, out):
        # inp usually: (bs, seq, hidden) for transformer linears
        # We want to accumulate stats across *token vectors*
        if inp.dim() == 2:
            # (tokens, hidden) -> pretend batch=1
            inp = inp.unsqueeze(0)
    
        if isinstance(self.layer, nn.Linear):
            # (bs, seq, hidden) -> (bs*seq, hidden)
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
    
            # inp: (n_tokens, in_features)
            inp = inp.to(torch.float32)
    
            n_tokens = inp.shape[0]
    
            # sum of squares per input feature
            cur = (inp ** 2).sum(dim=0)  # shape: (in_features,)
    
            # running mean over token-vectors
            if self.nsamples == 0:
                self.scaler_row = cur / n_tokens
                self.nsamples = n_tokens
            else:
                total = self.nsamples + n_tokens
                self.scaler_row = (self.scaler_row * self.nsamples + cur) / total
                self.nsamples = total
        else:
            # If you ever wrap non-linear layers, just skip safely
            return
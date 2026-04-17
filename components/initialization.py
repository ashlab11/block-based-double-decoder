import torch
import torch.nn as nn
import math
"""
Functions to initialize model weights.
"""

#Initialize norms
def _init_norms(module):
    nn.init.ones_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
        
def _get_embedding_weights(module):
    return getattr(module, "embedding", None).weight

def _get_num_layers(model):
    total = 0
    enc = getattr(model, "encoder_layers", None)
    recur = getattr(model, "recurrency_shape", None)
    dec = getattr(model, "decoder_layers", None)
    if recur is not None and enc is not None:
        raise ValueError("Model cannot have both encoder and recurrent layers")
    if enc is not None:
        total += len(enc)
    if recur is not None:
        avg_recurrencies = getattr(model, "avg_recurrencies", None)
        total += recur[0] + recur[1] * avg_recurrencies + recur[2]
    if dec is not None:
        total += len(dec)
    assert total > 0, "Model missing one or more layers"
    return total

#Initializes a model given lambda functions
def _initialize(model, weight_init, embed_init = None, gain = None):
    embedding_weights = _get_embedding_weights(model)
    if embed_init is not None:
        embed_init(embedding_weights)
    else:
        nn.init.normal_(embedding_weights, mean=0.0, std=0.02)
    for sub in model.modules():
        if isinstance(sub, nn.Linear):
            weight_init(sub.weight) if gain is None else weight_init(sub.weight, gain=gain)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.LayerNorm) or isinstance(sub, nn.RMSNorm):
            _init_norms(sub)

def _xavier_uniform(model):
    _initialize(model, nn.init.xavier_uniform_, gain=math.sqrt(2))

def _xavier_normal(model):
    _initialize(model, nn.init.xavier_normal_, gain=math.sqrt(2))

def _kaiming_uniform(model):
    _initialize(model, nn.init.kaiming_uniform_, gain=math.sqrt(2))

def _kaiming_normal(model):
    _initialize(model, nn.init.kaiming_normal_, gain=math.sqrt(2))

def _t5(model):
    dim = getattr(model, "dim", None)
    if dim is None:
        raise ValueError("Model needs to have a dim attribute")
    std = 1.0 / math.sqrt(dim)
    a, b = -2 * std, 2 * std
    init_fn = lambda w: nn.init.trunc_normal_(w, std=std, a=a, b=b)
    _initialize(model, init_fn, init_fn)

def _takase(model):
    dim = getattr(model, "dim", None)
    num_layers = _get_num_layers(model)
    if dim is None or num_layers is None:
        raise ValueError("Model missing one or more attributes")
    weight_std = math.sqrt(2.0 / (5.0 * dim))
    out_std = math.sqrt(1.0 / (5.0 * dim * num_layers))
    _initialize(model,
        lambda w: nn.init.trunc_normal_(w, std=weight_std, a=-2 * weight_std, b=2 * weight_std),
        lambda w: nn.init.trunc_normal_(w, std=out_std, a=-2 * out_std, b=2 * out_std))

def _ortho_takase(model):
    #Run takase first
    _takase(model)
    
    #Initialize all the recurrent blocks with orthogonal matrices
    if getattr(model, "recurrent_layers", None) is not None:
        layers_to_ortho = model.recurrent_layers
    else:
        layers_to_ortho = model.encoder_layers
    for layer in layers_to_ortho:
        for name, param in layer.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.orthogonal_(param)

INIT_STRATEGIES = {
    "xavier_uniform": _xavier_uniform,
    "xavier_normal": _xavier_normal,
    "kaiming_uniform": _kaiming_uniform,
    "kaiming_normal": _kaiming_normal,
    "t5": _t5,
    "takase": _takase,
    "ortho_takase": _ortho_takase,
}

def initialize_model(model, strategy):
    if strategy not in INIT_STRATEGIES:
        raise ValueError(f"Unknown init strategy: {strategy}")
    INIT_STRATEGIES[strategy](model)
    
def hidden_state_init(h, strategy):
    dim = h.shape[-1]
    if strategy == "xavier_uniform":
        nn.init.xavier_uniform_(h)
    elif strategy == "xavier_normal":
        nn.init.xavier_normal_(h)
    elif strategy == "kaiming_uniform":
        nn.init.kaiming_uniform_(h)
    elif strategy == "kaiming_normal":
        nn.init.kaiming_normal_(h)
    elif strategy == "takase":
        std = math.sqrt(2.0 / (5.0 * dim))
        a, b = -2 * std, 2 * std
        nn.init.trunc_normal_(h, std=std, a=a, b=b)
    elif strategy == "ortho_takase":
        nn.init.orthogonal_(h)
    else:
        raise ValueError(f"Unknown init strategy: {strategy}")
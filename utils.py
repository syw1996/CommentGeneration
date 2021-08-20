from transformers.utils.dummy_pt_objects import RagSequenceForGeneration
import torch
from torch import nn
from torch.nn import Parameter
from collections import defaultdict
import math

PAD, UNK, BOS, EOS = '[PAD]', '[UNK]', '[BOS]', '[EOS]'
LS, RS, SP = '[LS]', '[RS]', '[SPACE]'

def load_vocab(vocab_file):
    with open(vocab_file, 'r') as fr:
        id2token = [PAD, UNK, BOS, EOS, LS, RS, SP] + [line.strip().split('\t')[0] for line in fr.readlines()]
    token2id = {}
    for i, w in enumerate(id2token):
        token2id[w] = i
    return id2token, token2id

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x

def load_model(model, para_dict):
    state_dict = model.state_dict()
    for k, v in para_dict.items():
        if k not in state_dict:
            raise ValueError("key of para_dict not in model")
        state_dict[k] = v
    model.load_state_dict(state_dict)
    model.transformer.wte = nn.Embedding.from_pretrained(state_dict['transformer.wte.weight'])
    for k, v in model.state_dict().items():
        if not v.equal(state_dict[k]):
            raise ValueError("model para not match")
    return model

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.eps = eps
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)

def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_guyu_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._guyu_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._guyu_instance_id, key)

def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]

def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value

if __name__ == "__main__":
    id2token, _ = load_vocab('model/zh117M-L12-200G/vocab.txt')
    with open('model/transformers-gpt2-base/vocab.txt', 'w') as fw:
        fw.write('\n'.join(id2token))
    with open('model/transformers-gpt2-large/vocab.txt', 'w') as fw:
        fw.write('\n'.join(id2token))
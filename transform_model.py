import torch
from collections import OrderedDict

def convert_guyu_to_transformers(input_model_path, output_model_path, layer_num):
    guyu_model = torch.load(input_model_path)['model']
    transformers_model = OrderedDict()

    transformers_model['wte.weight'] = guyu_model['tok_embed.weight']
    transformers_model['wpe.weight'] = guyu_model['pos_embed.weights.weight']
    transformers_model['emb_layer_norm.weight'] = guyu_model['emb_layer_norm.weight']
    transformers_model['emb_layer_norm.bias'] = guyu_model['emb_layer_norm.bias']
    
    transformers_model['one_more.weight'] = guyu_model['one_more.weight']
    transformers_model['one_more.bias'] = guyu_model['one_more.bias']
    transformers_model['ln_f.weight'] = guyu_model['one_more_layer_norm.weight']
    transformers_model['ln_f.bias'] = guyu_model['one_more_layer_norm.bias']

    for i in range(layer_num):
        transformers_model['h.{}.ln_1.weight'.format(i)] = guyu_model['layers.{}.attn_layer_norm.weight'.format(i)]
        transformers_model['h.{}.ln_1.bias'.format(i)] = guyu_model['layers.{}.attn_layer_norm.bias'.format(i)]
        transformers_model['h.{}.ln_2.weight'.format(i)] = guyu_model['layers.{}.ff_layer_norm.weight'.format(i)]
        transformers_model['h.{}.ln_2.bias'.format(i)] = guyu_model['layers.{}.ff_layer_norm.bias'.format(i)]
        transformers_model['h.{}.attn.c_attn.weight'.format(i)] = guyu_model['layers.{}.self_attn.in_proj_weight'.format(i)].transpose(0, 1)
        transformers_model['h.{}.attn.c_attn.bias'.format(i)] = guyu_model['layers.{}.self_attn.in_proj_bias'.format(i)]
        transformers_model['h.{}.attn.c_proj.weight'.format(i)] = guyu_model['layers.{}.self_attn.out_proj.weight'.format(i)].transpose(0, 1)
        transformers_model['h.{}.attn.c_proj.bias'.format(i)] = guyu_model['layers.{}.self_attn.out_proj.bias'.format(i)]
        transformers_model['h.{}.mlp.c_fc.weight'.format(i)] = guyu_model['layers.{}.fc1.weight'.format(i)].transpose(0, 1)
        transformers_model['h.{}.mlp.c_fc.bias'.format(i)] = guyu_model['layers.{}.fc1.bias'.format(i)]
        transformers_model['h.{}.mlp.c_proj.weight'.format(i)] = guyu_model['layers.{}.fc2.weight'.format(i)].transpose(0, 1)
        transformers_model['h.{}.mlp.c_proj.bias'.format(i)] = guyu_model['layers.{}.fc2.bias'.format(i)]
    
    transformers_model_ = OrderedDict()
    for k, v in transformers_model.items():
        transformers_model_['transformer.' + k] = v

    transformers_model_['lm_head.weight'] = guyu_model['out_proj.weight']
    transformers_model_['lm_head.bias'] = guyu_model['out_proj.bias']

    torch.save(transformers_model_, output_model_path)

if __name__ == '__main__':
    convert_guyu_to_transformers('model/zh345m-L24-200G/zh345m-L24.m', 'model/transformers-gpt2-large/pytorch_model.bin', 24)
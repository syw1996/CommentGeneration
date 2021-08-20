import os
import torch
from transformers import TextGenerationPipeline

from modeling_gpt2_large import GPT2LMHeadModel
from tokenization_guyu import GuyuTokenizer

from ipdb import launch_ipdb_on_exception, set_trace

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

model = GPT2LMHeadModel.from_pretrained("model/transformers-gpt2-large/")
device = "cuda:0"
model.to(device)
print('loading is finished')
tokenizer = GuyuTokenizer.from_pretrained("model/transformers-gpt2-large/vocab.txt")
text_generator = TextGenerationPipeline(model, tokenizer)
with launch_ipdb_on_exception():
    while True:
        input_str = input().strip()
        input_ids = torch.LongTensor([tokenizer.encode(input_str, add_special_tokens=False)]).to(device)
        output_ids = model.generate(input_ids=input_ids, max_length=100, top_k=50, do_sample=True, bos_token_id=2, pad_token_id=0, eos_token_id=3)
        print('output: ' + ''.join(tokenizer.decode(output_ids[0]).split()))
        # print(text_generator(input_str, max_length=100, top_k=50, do_sample=True, bos_token_id=2, pad_token_id=0, eos_token_id=3))

import os
import torch
import yaml
import logging
from tqdm import tqdm

from modeling_gpt2 import GPT2LMHeadModel
from text_filtering.comment_filtering import filter_comment
from tokenization_guyu import GuyuTokenizer

logging.basicConfig(format="%(asctime)s - %(module)s : "
                           "%(funcName)s[%(filename)s:%(lineno)d] - "
                           "%(levelname)s: %(message)s",
                    datefmt="%m-%d %H:%M:%S %p",
                    level=logging.DEBUG)
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

class TextGenerator(object):
    def __init__(self, model_path, config):
        self.device = config["device"]
        model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model = model.to(self.device)
        tokenizer = GuyuTokenizer.from_pretrained("model/transformers-gpt2-base/vocab.txt")
        tokenizer.add_tokens(["SINGER"])
        self.tokenizer = tokenizer

        self.max_len = config["max_len"]
        self.num_sample = int(config["num_sample"])
        self.batch_size = config["batch_size"]
        self.k = config["k"] if "k" in config else 20
        self.p = config["p"] if "p" in config else 0.9
        self.bias = config["bias"] if "bias" in config else 0
        default_dic = dict(k=self.k, p=self.p, bias=self.bias)
        self.quantile_list = [default_dic] * self.max_len
        if isinstance(config["quantile"], list)\
           and len(config["quantile"]) != 0:
            q_cnt = 0
            for quantile in config["quantile"]:
                starting, end = quantile["starting"], quantile["end"]
                k, p, bias = quantile["k"], quantile["p"], quantile["bias"]
                for i in range(starting, end):
                    self.quantile_list[i] = dict(k=k, p=p, bias=bias)
                    q_cnt += 1
            if q_cnt != self.max_len:
                _res = self.max_len - q_cnt
                logging.warning("{} quantile is not initialized".format(_res))
        else:
            logging.warning("quntiles are not initialized!")
        self.reserved_idx = self.tokenizer.encode(config["reserved_tokens"], add_special_tokens=False)
        self.stop_ids = self.tokenizer.encode(config["stop_tokens"], add_special_tokens=False)
    

    def top_p_sampling(self, logits, k, p, vervose=False):
        ps, idx = torch.topk(logits, k=k)
        for i in range(k):
            if torch.sum(ps[:i]) >= p:
                ps = ps[:i]
                idx = idx[:i]
                break
        normalized_ps = ps / torch.sum(ps)
        sampled = torch.multinomial(normalized_ps, num_samples=1)
        sampled_idx = idx[sampled]
        if vervose:
            return sampled_idx, i,\
                   normalized_ps[sampled], ps[sampled]
            # idx, p_num, normalized_prob, unnormalize_prob
        else:
            return sampled_idx

    def top_k_sampling(self, logits, k,
                       replacement=True):
        ps, idx = torch.topk(logits, k=k)
        ps = ps / torch.sum(ps)
        if replacement:
            sampled = torch.multinomial(ps, num_samples=1)
        else:
            sampled = torch.multinomial(ps, num_samples=10, replacement=False)
        sampled_idx = idx[sampled]
        return sampled_idx

    @torch.no_grad()
    def adaptive_extended_sampling(self, input_str):
        past_key_values = None
        gen_list = [[] for _ in range(self.batch_size)]
        idx_list = [self.tokenizer.encode(list(input_str) + ["[RS]"], add_special_tokens=False) for _ in range(self.batch_size)]
        x = torch.LongTensor(idx_list)
        x = x.to(self.device)
        discound_idx = None
        for l_idx in range(self.max_len):
            attention_mask = x.new_ones(x.shape, dtype=torch.long).to(self.device)
            if l_idx == 0:
                input_ids = x
            else:
                input_ids = x[:, -1].unsqueeze(1)
            
            outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, 
                                                attention_mask=attention_mask)
            lm_logits, past_key_values = outputs[0], outputs[1]
            probs = torch.softmax(lm_logits, dim=-1)
            for stop_id in self.stop_ids:
                probs[:, -1, stop_id] = 0
            next_tk, next_idx = [], []
            for batch_idx in range(self.batch_size):
                logits = probs[batch_idx, -1]
                if discound_idx is not None:
                    logits[discound_idx[batch_idx]] = \
                        logits[discound_idx[batch_idx]] * 0.01
                quantie_dic = self.quantile_list[l_idx]
                k, p = quantie_dic["k"], quantie_dic["p"]
                bias = quantie_dic["bias"]
                logits = logits[bias:]
                if l_idx != 0:
                    sampled_idx = self.top_p_sampling(logits, k, p)
                    sampled_idx = sampled_idx + bias
                    next_tk.append(self.tokenizer.decode([sampled_idx.item()]))
                    next_idx.append(sampled_idx.item())
                elif batch_idx == 0:
                    sampled_idx = self.top_k_sampling(logits, k=k, replacement=False)
                    sampled_idx = sampled_idx + bias
                    next_tk = [self.tokenizer.decode([idx]) for idx in sampled_idx.tolist()]
                    next_idx = sampled_idx.tolist()
                else:
                    continue
            eliminate_idx = map(lambda x: 29 if x in self.reserved_idx else x, next_idx)
            eliminate_idx = list(eliminate_idx)
            if l_idx == 0:
                discound_idx = x.new(eliminate_idx).unsqueeze(1)
            else:
                inc_discound_idx = x.new(eliminate_idx).unsqueeze(-1)
                discound_idx = torch.cat([discound_idx, inc_discound_idx], dim=1)
            for i, tk in enumerate(next_tk):
                gen_list[i].append(tk)
                idx_list[i].append(next_idx[i])
            x = torch.LongTensor(idx_list)
            x = x.to(self.device)

        returned_list = [''.join(sent) for sent in gen_list]

        return returned_list


def single_test(config_path, input_fn):
    with open(config_path) as f:
        config = yaml.load(f, yaml.CLoader)
    generator = TextGenerator(config["model_path"], config)
    output_fn = config["output_path"]
    with open(input_fn, "r") as fr, open(output_fn, "w") as fw, open("gen/only_review_loss_results/dirty.txt", "w") as fd:
        for line in tqdm(fr.readlines()):
            fw.write(line.strip() + "\n" + "=" * 30 + "\n")
            singer_name, gender, lyric = line.strip().split("<sep>")
            lyric = lyric.replace("\\n", "。").replace(" ", "")
            cnt = 0
            while cnt < 10:
                gen_list = generator.adaptive_extended_sampling(lyric)
                fd.write("\n".join(gen_list))
                fd.flush()
                clean_comments = filter_comment(gen_list, gender)
                for comment in clean_comments:
                    fw.write(comment + "\n")
                cnt += len(clean_comments)
            fw.write("\n")
            fw.flush()


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    with launch_ipdb_on_exception():
        single_test("./generate_config.yaml",
                    "./data/lyrics/sample_lyric_sections.txt")
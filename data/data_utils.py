import os
import pickle
from typing import Dict

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, logging, DataCollatorForLanguageModeling
from tqdm import tqdm
from typing import Dict

import sys 
sys.path.append("..") 
from tokenization_guyu import GuyuTokenizer

logger = logging.get_logger(__name__)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            # lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            self.examples = [list(map(int, line.split())) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # batch_encoding = tokenizer(lines, add_special_tokens=False, truncation=False)
        # self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = LineByLineTextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=1024)
     
    test_dataset = LineByLineTextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=1024)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


def merge_lyric2review():
    merged_lines = []
    for i in range(5):
        with open("processed_reviews/processed_review_{}.txt".format(i), "r")  as fr:
            for line in fr.readlines():
                _, _, lyric, review = line.strip().split("<sep>")
                lyric = '。'.join(lyric.split("\\n"))
                merged_lines.append(lyric + "[RS]" + review + "[EOS]")
    with open("processed_reviews/lyric_to_review.txt", "w") as fw:
        fw.write("\n".join(merged_lines))


def build_seq2seq_train_test_file():
    from sklearn.model_selection import train_test_split
    with open("processed_reviews/lyric_to_review.txt", "r")  as fr:
        src_data, tgt_data = [], []
        for line in fr.readlines():
            lyric, view = line.strip().split("[RS]")
            src_data.append(lyric)
            # Seq2SeqDataCollator中含有shift_tokens_right，会右移decoder的input_ids并在头部增加bos_id
            tgt_data.append(view)
    src_train, src_test, tgt_train, tgt_test = train_test_split(src_data, tgt_data, test_size=0.01, random_state=123) 
    with open("processed_reviews/seq2seq/train.source", "w") as fw:
        fw.write("\n".join(src_train) + "\n")
    with open("processed_reviews/seq2seq/test.source", "w") as fw:
        fw.write("\n".join(src_test) + "\n")
    with open("processed_reviews/seq2seq/train.target", "w") as fw:
        fw.write("\n".join(tgt_train) + "\n")
    with open("processed_reviews/seq2seq/test.target", "w") as fw:
        fw.write("\n".join(tgt_test) + "\n")


def build_gpt2_train_test_file(tokenizer):
    from sklearn.model_selection import train_test_split
    with open("processed_reviews/lyric_to_review.txt", "r")  as fr:
        data = [line.strip() for line in fr.readlines()]
        train, test = train_test_split(data, test_size=0.01, random_state=123) 
    with open("processed_reviews/train_dataset.txt", "w") as fw:
        fw.write("\n".join(train) + "\n")
    with open("processed_reviews/test_dataset.txt", "w") as fw:
        fw.write("\n".join(test) + "\n")
    def obtain_tokenized_file(lines):
        tokenized_data = []
        for line in tqdm(lines):
            token_ids = list(map(str, tokenizer.encode(line, add_special_tokens=False)))
            if len(token_ids) <= 512:
                tokenized_data.append(" ".join(token_ids))
        return tokenized_data
    with open("processed_reviews/tokenized_train_dataset.txt", "w") as fw:
        fw.write("\n".join(obtain_tokenized_file(train)) + "\n")
    with open("processed_reviews/tokenized_test_dataset.txt", "w") as fw:
        fw.write("\n".join(obtain_tokenized_file(test)) + "\n")



if __name__ == "__main__":
    # merge_lyric2review()
    """
    tokenizer = GuyuTokenizer.from_pretrained("../model/transformers-gpt2-base/vocab.txt")
    tokenizer.add_tokens(['SINGER'])
    build_gpt2_train_test_file(tokenizer)
    """
    build_seq2seq_train_test_file()
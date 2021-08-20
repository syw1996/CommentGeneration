import os
from typing import Dict

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, logging, DataCollatorForLanguageModeling
from tqdm import tqdm

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


def build_train_test_file(tokenizer):
    from sklearn.model_selection import train_test_split
    with open("processed_reviews/lyric_to_review.txt", "r")  as fr:
        # data = [line.strip() for line in fr.readlines()]
        data = []
        for line in tqdm(fr.readlines()):
            token_ids = list(map(str, tokenizer.encode(line, add_special_tokens=False)))
            if len(token_ids) <= 512:
                data.append(" ".join(token_ids))
    train, test = train_test_split(data, test_size=0.05) 
    with open("processed_reviews/tokenized_train_dataset.txt", "w") as fw:
        fw.write("\n".join(train))
    with open("processed_reviews/tokenized_test_dataset.txt", "w") as fw:
        fw.write("\n".join(test))


if __name__ == "__main__":
    # merge_lyric2review()
    tokenizer = GuyuTokenizer.from_pretrained("../model/transformers-gpt2-base/vocab.txt")
    tokenizer.add_tokens(['SINGER'])
    build_train_test_file(tokenizer)
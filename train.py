import argparse, os
import time
import deepspeed
from transformers import Trainer, TrainingArguments

from data.data_utils import load_dataset
from tokenization_guyu import GuyuTokenizer
from modeling_gpt2 import GPT2LMHeadModel


class Config():
    def __init__(self):
        self.train_data = "./data/processed_reviews/tokenized_train_dataset.txt"
        self.test_data = "./data/processed_reviews/tokenized_test_dataset.txt"
        self.model_dir = "./model/transformers-gpt2-base/"
        self.deepspeed = "./ds_config.json"
        self.per_device_batch_size = 16
        self.warmup_steps = 10000
        self.lr = 0.0003608
        self.eval_steps = 5000
        self.save_steps = 5000
        self.epoch = 100
        self.save_dir = "./model/lyric_to_review"
        self.gpus = "2,3,4,5"


def run(config, add_tokens_lst):
    train_path = config.train_data
    test_path = config.test_data
    tokenizer = GuyuTokenizer.from_pretrained(os.path.join(config.model_dir, "vocab.txt"))
    tokenizer.add_tokens(add_tokens_lst)

    start = time.time()
    train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
    print("data loading uses {} seconds".format(time.time() - start))

    model = GPT2LMHeadModel.from_pretrained(config.model_dir)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=config.save_dir, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        do_eval=True,
        evaluation_strategy="steps",
        adam_epsilon=1e-9,
        num_train_epochs=config.epoch, # number of training epochs
        per_device_train_batch_size=config.per_device_batch_size, # batch size for training
        per_device_eval_batch_size=config.per_device_batch_size,  # batch size for evaluation
        learning_rate=config.lr,
        eval_steps=config.eval_steps, # Number of update steps between two evaluations.
        save_steps=config.save_steps, # after # steps model is saved 
        warmup_steps=config.warmup_steps,# number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        fp16=False,
        # fp16_backend="amp",
        # deepspeed=config.deepspeed,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    run(config, ["SINGER"])
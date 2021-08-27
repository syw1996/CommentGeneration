import os
import deepspeed

from dataclasses import dataclass
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from data.seq2seq_utils import (
    DataTrainingArguments, 
    Seq2SeqDataset, 
    Seq2SeqDataCollator, 
    build_compute_metrics_fn,
)
from modeling_encoder_decoder import EncoderDecoderModel
from tokenization_guyu import GuyuTokenizer

@dataclass
class Config:
    data_dir = "./data/processed_reviews/seq2seq/"
    encoder_model_dir = "./model/transformers-gpt2-base/"
    decoder_model_dir = "./model/transformers-gpt2lmhead-base/"
    deepspeed = "./ds_config.json"
    per_device_batch_size = 16
    warmup_steps = 10000
    lr = 0.0003608
    eval_steps = 5000
    save_steps = 5000
    logging_steps = 1
    epoch = 50
    save_dir = "./model/lyric_to_review_seq2seq"
    gpus = "2,3"


def run(config, add_tokens_lst):
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(config.encoder_model_dir, config.decoder_model_dir)
    tokenizer = GuyuTokenizer.from_pretrained(os.path.join(config.decoder_model_dir, "vocab.txt"))
    tokenizer.add_tokens(add_tokens_lst)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Get datasets
    data_args = DataTrainingArguments(data_dir=config.data_dir)
    train_dataset = Seq2SeqDataset(
        tokenizer,
        type_path="train",
        data_dir=data_args.data_dir,
        n_obs=data_args.n_train,
        max_target_length=data_args.max_target_length,
        max_source_length=data_args.max_source_length,
        prefix=model.config.prefix or "",
    )
    eval_dataset = Seq2SeqDataset(
        tokenizer,
        type_path="test",
        data_dir=data_args.data_dir,
        n_obs=data_args.n_val,
        max_target_length=data_args.val_max_target_length,
        max_source_length=data_args.max_source_length,
        prefix=model.config.prefix or "",
    )

    compute_metrics_fn = build_compute_metrics_fn(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.save_dir, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        adam_epsilon=1e-4,
        num_train_epochs=config.epoch, # number of training epochs
        per_device_train_batch_size=config.per_device_batch_size, # batch size for training
        per_device_eval_batch_size=config.per_device_batch_size,  # batch size for evaluation
        learning_rate=config.lr,
        eval_steps=config.eval_steps, # Number of update steps between two evaluations.
        save_steps=config.save_steps, # after # steps model is saved 
        warmup_steps=config.warmup_steps,# number of warmup steps for learning rate scheduler
        prediction_loss_only=True, # if use bleu metirc, set False
        logging_steps=config.logging_steps,
        fp16=True,
        fp16_opt_level="O1",
        fp16_backend="amp",
        # deepspeed=config.deepspeed,
        # no_cuda=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        # data_args=data_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(
            tokenizer, data_args, model.config.decoder_start_token_id, training_args.tpu_num_cores
        ),
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    from ipdb import launch_ipdb_on_exception
    with launch_ipdb_on_exception():
        run(config, ["SINGER"])
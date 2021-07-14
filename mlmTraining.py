import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse

def mlm_params():

    parser = argparse.ArgumentParser()
    
    # model params
    parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
    
    # training params
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--l2', type=float, default=0.003)

    # data params
    parser.add_argument('--trainDir', type=str)
    parser.add_argument('--valDir', type=str)
    
    # misc. params
    parser.add_argument('--paramDir', type=str)
    parser.add_argument('--logDir', type=str)
    
    return parser.parse_args()

args = mlm_params()


tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer)
model = DistilBertForMaskedLM.from_pretrained(args.encoder)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.trainDir,
    block_size=128,
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path= args.valDir,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir= args.paramDir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batchSize,
    save_steps=20,
    evaluation_strategy="steps",
    fp16=True,
    weight_decay=args.l2,
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit = 1,
    logging_dir= args.logDir,
    metric_for_best_model="eval_loss",
    prediction_loss_only=True
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
    )

trainer.train()

trainer.save_model(args.paramDir + '/best/')
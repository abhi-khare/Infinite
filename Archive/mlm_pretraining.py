import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse
parser = argparse.ArgumentParser()

###################################################################################################################
parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')

parser.add_argument('--block_size', type=int, default=128)
parser.add_argument('--mlm_prob', type=float, default=0.15)
parser.add_argument('--train_dir',type=str)
parser.add_argument('--val_dir',type=str)
parser.add_argument('--export_dir', type=str)
parser.add_argument('--log_dir',type=str,default='logs/mlm_training')

# training parameters
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--save_step',type=int,default=5000)
parser.add_argument('--weight_decay',type=float,default=0.003)


args = parser.parse_args()

###################################################################################################################

tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer_weights)
model = DistilBertForMaskedLM.from_pretrained(args.model_weights)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.train_dir,
    block_size=args.block_size,
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.val_dir,
    block_size=args.block_size,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
)

training_args = TrainingArguments(
    output_dir=args.export_dir,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    save_steps=args.save_step,
    evaluation_strategy="steps",
    fp16=True,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit = 1,
    logging_dir=args.log_dir,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model(args.export_dir)


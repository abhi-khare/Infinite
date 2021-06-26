import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


basePath = './'

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=basePath + 'data/SNIPS/experiments/train/adversarialAdaptiveBG_train.tsv',
    block_size=128,
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path= basePath + 'data/SNIPS/experiments/dev/aadversarialAdaptiveDev_BG.tsv',
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir= basePath + 'bin/SNIPS/MLM_Adver_Adap_BG/',
    num_train_epochs=10,
    per_device_train_batch_size=128,
    save_steps=20,
    evaluation_strategy="steps",
    fp16=True,
    weight_decay=0.003,
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit = 1,
    logging_dir= basePath + 'logs/SNIPS/MLM_Adver_Adap_BG/',
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

trainer.save_model(basePath + 'bin/SNIPS/MLM_Adver_Adap_BG/best/')
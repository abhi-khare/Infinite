# baseline jointBert model on 4 languages

#python IC_NER_Training.py --train_dir ./data/splits/rich/multi-train.tsv --val_dir ./data/splits/rich/multi-dev.tsv --exp_name jointBert --slots_dropout 0.2055 --weight_decay 0.0020 --intent_dropout 0.1529 --intent_lr 0.0000806 --encoder_lr 0.0000315 --slots_lr 0.000309

# isolated jointBert tuning on en dataset.
python jointBertTuning.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv 
# baseline jointBert model on en dataset

python IC_NER_Training.py --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name jointBert_EN --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823

# isolated jointBert tuning on en dataset.
#python jointBertTuning.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv 
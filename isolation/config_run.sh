## Experiment config file



# Experiment: intent contrastive pre training

# Exp 1 triplet margin loss margin = 2.0
python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_2 --exp_name ./logs/ICPT_TML_2 --margin 2.0
python IC_NER_Training.py --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name jointBert_EN --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823










# baseline jointBert model on en dataset

#python IC_NER_Training.py --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name jointBert_EN --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823

# isolated jointBert tuning on en dataset.
#python jointBertTuning.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv 


#python intentContrastiveTuning.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --device cpu 

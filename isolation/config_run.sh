## Experiment config file

# baseline: jointBert
#python IC_NER_Training.py --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name jointBert_EN --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823


# Experiment: intent contrastive pre training

# Exp 1 triplet margin loss margin = 2.0
#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_2 --exp_name ./logs/ICPT_TML_2 --margin 2.0
#python IC_NER_Training.py --model_name ./bin/ICPT_TML_2 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_TML_2 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823

# Exp 2 triplet margin loss margin = 5.0
python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_5 --exp_name ./logs/ICPT_TML_5 --margin 5.0
python IC_NER_Training.py --model_name ./bin/ICPT_TML_5 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_TML_5 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823

# Exp 3 triplet margin loss margin = 10.0
python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_10 --exp_name ./logs/ICPT_TML_10 --margin 10.0
python IC_NER_Training.py --model_name ./bin/ICPT_TML_10 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_TML_10 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000074857785 --slots_lr 0.000744823

# Exp 4 contrastive loss 

3-4 variants of this.

# Exp 5 INFONICE loss

3-4 variants of this.


# Experiment: intent and slots contrastive pre training


## Experiment config file

# baseline: multi-lingual jointBert
#python IC_NER_Training.py --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name jointBert_multi --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Experiment: ICPT + triplet margin loss

# Exp 1 triplet margin loss margin = 2.0
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_TML_2 --exp_name ./logs/multi_ICPT_TML_2 --margin 2.0 --loss TML
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_TML_2 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_TML_2 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 2 triplet margin loss margin = 5.0
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_TML_5 --exp_name ./logs/multi_ICPT_TML_5 --margin 5.0 --loss TML
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_TML_5 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_TML_5 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 3 triplet margin loss margin = 0.05 (base)
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_TML_0_05 --exp_name ./logs/multi_ICPT_TML_0_05 --margin 0.05 --loss TML
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_TML_0_05 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_TML_0_05 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823



# Experiment: ICPT + contrastive loss

# Exp 1 triplet margin loss margin = 2.0
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_CONTRA_2 --exp_name ./logs/multi_ICPT_CONTRA_2 --neg_margin 2.0 --loss CONTRA
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_CONTRA_2 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_CONTRA_2 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 2 triplet margin loss margin = 5.0
python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_CONTRA_5 --exp_name ./logs/multi_ICPT_CONTRA_5 --neg_margin 5.0 --loss CONTRA
python IC_NER_Training.py --model_name ./bin/multi_ICPT_CONTRA_5 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_CONTRA_5 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.0000 --slots_lr 0.000744823

# Exp 2 triplet margin loss margin = 1.0 (base)
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_CONTRA_1 --exp_name ./logs/multi_ICPT_CONTRA_1 --neg_margin 1.0 --loss CONTRA
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_CONTRA_1 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_CONTRA_1 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Experiment: ICPT + InfoNCE loss

# Exp 2 triplet margin loss margin = 5.0
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_NXT_0_07 --exp_name ./logs/multi_ICPT_NXT_0_07 --temperature 0.07
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_NXT_0_07 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_NXT_0_07 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 3 triplet margin loss margin = 0.05 (base)
#python intentContrastive.py --train_dir ./data/multi-train.tsv --val_dir ./data/multi-dev.tsv --model_export ./bin/multi_ICPT_NXT_0_5 --exp_name ./logs/multi_ICPT_NXT_0_5 --temperature 0.5
#python IC_NER_Training.py --model_name ./bin/multi_ICPT_NXT_0_5 --train_dir ./data/multi-train.tsv --val_dir ./data/  --exp_name ./logs/multi_IC_NER_ICPT_NXT_0_5 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823




# margin loss margin = 2.0
#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_2 --exp_name ./logs/ICPT_TML_2 --margin 2.0
#python IC_NER_Training.py --model_name ./bin/ICPT_TML_2 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_TML_2 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 2 triplet margin loss margin = 5.0
#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_5 --exp_name ./logs/ICPT_TML_5 --margin 5.0
#python IC_NER_Training.py --model_name ./bin/ICPT_TML_5 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_TML_5 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 3 triplet margin loss margin = 10.0
#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_TML_10 --exp_name ./logs/ICPT_TML_10 --margin 10.0
#python IC_NER_Training.py --model_name ./bin/ICPT_TML_10 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_TML_10 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 4-5 contrastive loss 

#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_NXT_0_07 --exp_name ./logs/ICPT_NXT_0_07 --temperature 0.07
#python IC_NER_Training.py --model_name ./bin/ICPT_NXT_0_07 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_NXT_0_07 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_NXT_0_5 --exp_name ./logs/ICPT_NXT_0_5 --temperature 0.5
#python IC_NER_Training.py --model_name ./bin/ICPT_NXT_0_5 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_NXT_0_5 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

# Exp 6-7 INFONICE loss

#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_CONTRA_0_1 --exp_name ./logs/ICPT_CONTRA_0_1 --neg_margin 1.0
#python IC_NER_Training.py --model_name ./bin/ICPT_CONTRA_0_1 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_CONTRA_0_1 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

#python intentContrastive.py --train_dir ./data/train_EN.tsv --val_dir ./data/dev_EN.tsv --model_export ./bin/ICPT_CONTRA_0_2 --exp_name ./logs/ICPT_CONTRA_0_2 --neg_margin 2.0
#python IC_NER_Training.py --model_name ./bin/ICPT_CONTRA_0_2 --train_dir ./data/train_EN.tsv --val_dir ./data/  --exp_name ./logs/IC_NER_ICPT_CONTRA_0_2 --slots_dropout 0.1949101 --weight_decay 0.0022488 --intent_dropout 0.25895432 --intent_lr 0.00097826 --encoder_lr 0.000014857785 --slots_lr 0.000744823

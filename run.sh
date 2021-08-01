
# HP Tuning ATIS and SNIPS dataset
env TRANSFORMERS_OFFLINE=1  python jointBertTuner.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_count 18 --slots_count 120 --dataset ATIS --logging_dir ./logs/HPT_ATIS --epoch 20  2>&1 | tee ./logs/HPT_ATIS.txt
env TRANSFORMERS_OFFLINE=1  python jointBertTuner.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_count 8 --slots_count 72 --dataset SNIPS --logging_dir ./logs/HPT_SNIPS 2>&1 | tee ./logs/HPT_SNIPS.txt

# baseline jointBert experiment
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --intent_count 18 --slots_count 120 --dataset ATIS --param_save_dir ./bin/BASELINE_ATIS/ --logging_dir ./logs/BASELINE_ATIS/ 
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --intent_count 8 --slots_count 72 --dataset SNIPS --param_save_dir ./bin/BASELINE_SNIPS/ --logging_dir ./logs/BASELINE_SNIPS/ --epoch 6

# Adversarial Training MC noise experiments for ATIS and SNIPS
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --intent_count 18 --slots_count 120 --dataset ATIS --param_save_dir ./bin/AT_MC_ATIS/ --logging_dir ./logs/AT_MC_ATIS/ --mode AT --noise_type MC --epoch 50
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --intent_count 8 --slots_count 72 --dataset SNIPS --param_save_dir ./bin/AT_MC_SNIPS/ --logging_dir ./logs/AT_MC_SNIPS/ --mode AT --noise_type MC --epoch 30

# Adversarial Training BG noise experiments for ATIS and SNIPS
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --intent_count 18 --slots_count 120 --dataset ATIS --param_save_dir ./bin/AT_BG_ATIS/ --logging_dir ./logs/AT_BG_ATIS/ --mode AT --noise_type BG --epoch 50
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --intent_count 8 --slots_count 72 --dataset SNIPS --param_save_dir ./bin/AT_BG_SNIPS/ --logging_dir ./logs/AT_BG_SNIPS/ --mode AT --noise_type BG --epoch 30


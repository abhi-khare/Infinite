
# HP Tuning ATIS and SNIPS dataset
# env TRANSFORMERS_OFFLINE=1  python jointBertTuner.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_count 18 --slots_count 120 --dataset ATIS --logging_dir ./logs/HPT_ATIS --epoch 20  2>&1 | tee ./logs/HPT_ATIS.txt
# env TRANSFORMERS_OFFLINE=1  python jointBertTuner.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_count 8 --slots_count 72 --dataset SNIPS --logging_dir ./logs/HPT_SNIPS 2>&1 | tee ./logs/HPT_SNIPS.txt

# baseline jointBert experiment
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_dropout --slots_dropout --lr --intent_count 18 --slots_count 120 --dataset ATIS --param_save_dir ./bin/BASELINE_ATIS/ --logging_dir ./logs/BASELINE_ATIS/ 
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_dropout --slots_dropout --lr --intent_count 8 --slots_count 72 --dataset SNIPS --param_save_dir ./bin/BASELINE_SNIPS/ --logging_dir ./logs/BASELINE_SNIPS/ 

# Adversarial Training MC noise experiments for ATIS and SNIPS
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_dropout --slots_dropout --lr --intent_count 18 --slots_count 120 --dataset ATIS --param_save_dir ./bin/BASELINE_ATIS/ --logging_dir ./logs/BASELINE_ATIS/ --mode AT --noise_type MC
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_dropout --slots_dropout --lr --intent_count 8 --slots_count 72 --dataset SNIPS --param_save_dir ./bin/BASELINE_SNIPS/ --logging_dir ./logs/BASELINE_SNIPS/ --mode AT --noise_type MC

# Adversarial Training BG noise experiments for ATIS and SNIPS
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_dropout --slots_dropout --lr --intent_count 18 --slots_count 120 --dataset ATIS --param_save_dir ./bin/BASELINE_ATIS/ --logging_dir ./logs/BASELINE_ATIS/ --mode AT --noise_type BG
env TRANSFORMERS_OFFLINE=1  python jointBertTrainer.py --train_dir ./data/SNIPS/experiments/train/clean/train.tsv --val_dir ./data/SNIPS/experiments/dev/clean/dev.tsv --intent_dropout --slots_dropout --lr --intent_count 8 --slots_count 72 --dataset SNIPS --param_save_dir ./bin/BASELINE_SNIPS/ --logging_dir ./logs/BASELINE_SNIPS/ --mode AT --noise_type BG

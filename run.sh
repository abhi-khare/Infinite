
# HP Tuning ATIS dataset
# env TRANSFORMERS_OFFLINE=1  python jointBertTuner.py --train_dir ./data/ATIS/experiment/train/clean/train.tsv --val_dir ./data/ATIS/experiment/dev/clean/dev.tsv --intent_count 18 --slots_count 120 --dataset ATIS --logging_dir ./ 

# HP Tuning SNIPS dataset
env TRANSFORMERS_OFFLINE=1  python jointBertTuner.py --train_dir ./data/SNIPS/experiment/train/clean/train.tsv --val_dir ./data/SNIPS/experiment/dev/clean/dev.tsv --intent_count 8 --slots_count 72 --dataset SNIPS --logging_dir ./ --desc SNIPS_BASELINE_HP

# HP Tuning ATIS dataset
python jointBertTuner.py --train_dir ./data/ATIS/experiment/train/clean/train.tsv --val_dir ./data/ATIS/experiment/dev/clean/dev.tsv --intent_count 18 --slots_count 120 --dataset ATIS --logging_dir ./ 

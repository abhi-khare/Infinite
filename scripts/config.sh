# Hyper parameter tuning command 
env TRANSFORMERS_OFFLINE=1  python ICTuner.py --train_dir ./data/ATIS/experiments/train/clean/train.tsv --val_dir ./data/ATIS/experiments/dev/clean/dev.tsv --intent_count 18  --logging_dir ./logs/HPT_ATIS --epoch 20  2>&1 | tee ./logs/HPT_ATIS.txt

# baseline bert training and test
#ATIS
env TRANSFORMERS_OFFLINE=1  python ICTrainer.py --train_dir ./data/ATIS/experiments/clean/train/train.tsv --val_dir ./data/ATIS/experiments/clean/dev/dev.tsv --num_class 18 --intent_dropout 0.24864368973706322 --intent_hidden 239 --lr 0.000027072493121027496 --param_save_dir ./bin/baseline_bert/ATIS --freeze_args True --logging_dir ./logs/baseline_bert/ATIS --exp_num baseline_bert
env TRANSFORMERS_OFFLINE=1  python ICTesting.py --test_dir ./ --num_class 18 --intent_dropout 0.24864368973706322 --intent_hidden 239 --lr 0.000027072493121027496  --freeze_args True --dataset ATIS --model_dir bin/baseline_bert/ATIS/ --batch_size 19

#SNIPS
env TRANSFORMERS_OFFLINE=1  python ICTrainer.py --train_dir ./data/SNIPS/experiments/clean/train/train.tsv --val_dir ./data/SNIPS/experiments/clean/dev/dev.tsv --num_class 8 --intent_dropout 0.21303972657269768 --intent_hidden 44 --lr 0.00003127820114422006 --param_save_dir ./bin/baseline_bert/SNIPS --freeze_args True --logging_dir ./logs/baseline_bert/SNIPS --exp_num baseline_bert
env TRANSFORMERS_OFFLINE=1  python ICTesting.py --test_dir ./ --num_class 8 --intent_dropout 0.21303972657269768 --intent_hidden 44 --lr 0.00003127820114422006  --freeze_args True --dataset SNIPS --model_dir bin/baseline_bert/SNIPS/ --batch_size 10

#TOD 
env TRANSFORMERS_OFFLINE=1  python ICTrainer.py --train_dir ./data/TOD/experiments/clean/train/train.tsv --val_dir ./data/TOD/experiments/clean/dev/dev.tsv --num_class 13 --intent_dropout 0.3025385895139513 --intent_hidden 35 --lr 0.00001184100185472858 --param_save_dir ./bin/baseline_bert/TOD --freeze_args True --logging_dir ./logs/baseline_bert/TOD --exp_num baseline_bert
env TRANSFORMERS_OFFLINE=1  python ICTesting.py --test_dir ./ --num_class 13 --intent_dropout 0.3025385895139513 --intent_hidden 35 --lr 0.00001184100185472858  --freeze_args True --dataset TOD --model_dir bin/baseline_bert/TOD/ --batch_size 37

# back translation testing

# ATIS RU and DE

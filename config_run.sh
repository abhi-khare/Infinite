#python IC_NER_Training.py --train_dir ./data/splits/multi-train.tsv --val_dir ./data/splits/multi-dev.tsv --exp_name exp1 --slots_dropout 0.2442 --intent_dropout 0.3803 --encoder_lr 0.000108 --rest_lr 0.000329

#python mlm_pretraining.py --train_dir ./data/splits/text/multi-train_text.txt --val_dir ./data/splits/text/multi-dev_text.txt --export_dir ./bin

python IC_NER_Training.py --train_dir ./data/splits/multi-train.tsv --val_dir ./data/splits/multi-dev.tsv --exp_name exp1 --slots_dropout 0.2442 --intent_dropout 0.3803 --encoder_lr 0.000108 --rest_lr 0.000329 --model_weights ./bin/checkpoint-5000
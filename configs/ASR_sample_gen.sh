# ATIS
env TRANSFORMERS_OFFLINE=1 python speech2text.py --noise_data_path data/noise_resources/background_noise/ --sample_path data/ATIS/raw/test/ASR_voice_samples/ --out_path data/ATIS/experiments/ASR_BGnoise/test --dataset_path data/ATIS/experiments/clean/test/test.tsv

# SNIPS
env TRANSFORMERS_OFFLINE=1 python speech2text.py --noise_data_path data/noise_resources/background_noise/ --sample_path data/SNIPS/raw/test/ASR_voice_samples/ --out_path data/SNIPS/experiments/ASR_BGnoise/test --dataset_path data/SNIPS/experiments/clean/test/test.tsv

# TOD
env TRANSFORMERS_OFFLINE=1 python speech2text.py --noise_data_path data/noise_resources/background_noise/ --sample_path data/TOD/raw/test/ASR_voice_samples/ --out_path data/TOD/experiments/ASR_BGnoise/test --dataset_path data/TOD/experiments/clean/test/test.tsv
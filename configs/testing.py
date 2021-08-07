# testing configs

# baseline test results for ATIS and SNIPS

env TRANSFORMERS_OFFLINE=1 python modelTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/BASELINE_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python modelTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/BASELINE_SNIPS/ --test_path /efs-storage/Infinite/



# Adversarial training test results for ATIS and SNIPS [MC Noise]
env TRANSFORMERS_OFFLINE=1 python modelTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/AT_MC_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python modelTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/AT_MC_SNIPS/ --test_path /efs-storage/Infinite/

# Adversarial training test results for ATIS and SNIPS [BG Noise]
env TRANSFORMERS_OFFLINE=1 python modelTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/AT_BG_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python modelTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/AT_BG_SNIPS/ --test_path /efs-storage/Infinite/


# HierCon test results for ATIS and SNIPS [MC Noise]
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/CT_MC_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/CT_MC_SNIPS/ --test_path /efs-storage/Infinite/

# HierCon test results for ATIS and SNIPS [BG Noise]
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/CT_BG_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/CT_BG_SNIPS/ --test_path /efs-storage/Infinite/

# HierCon + AT test results for ATIS and SNIPS [MC Noise]
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/CT_AT_MC_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/CT_AT_MC_SNIPS/ --test_path /efs-storage/Infinite/

# HierCon + AT test results for ATIS and SNIPS [BG Noise]
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 18 --slots_count 120 --dataset ATIS --intent_dropout 0.4959487455371492 --slots_dropout 0.4023767420439718 --lr 0.00003436854631596889 --base_dir /efs-storage/Infinite/bin/CT_BG_ATIS/ --test_path /efs-storage/Infinite/
env TRANSFORMERS_OFFLINE=1 python hierConTesting.py --intent_count 8 --slots_count 72 --dataset SNIPS --intent_dropout 0.44971200311949866 --slots_dropout 0.31526667279678505 --lr 0.000056253710225502357 --base_dir /efs-storage/Infinite/bin/CT_BG_SNIPS/ --test_path /efs-storage/Infinite/





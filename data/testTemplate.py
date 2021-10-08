def test_template(test, basePath):

    if test == "ATIS":
        test_50BG = [
            basePath + "data/ATIS/experiment/test/NoiseBG/50per/test_01.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/50per/test_02.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/50per/test_03.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/50per/test_04.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/50per/test_05.tsv",
        ]

        test_25BG = [
            basePath + "data/ATIS/experiment/test/NoiseBG/25per/test_01.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/25per/test_02.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/25per/test_04.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/25per/test_05.tsv",
        ]

        test_75BG = [
            basePath + "data/ATIS/experiment/test/NoiseBG/75per/test_01.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/75per/test_02.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/75per/test_03.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/75per/test_04.tsv",
            basePath + "data/ATIS/experiment/test/NoiseBG/75per/test_05.tsv",
        ]

        test_20OOC = [
            basePath + "data/ATIS/experiment/test/NoiseOOC/20per/test_01.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/20per/test_02.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/20per/test_03.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/20per/test_04.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/20per/test_05.tsv",
        ]

        test_40OOC = [
            basePath + "data/ATIS/experiment/test/NoiseOOC/40per/test_01.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/40per/test_02.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/40per/test_03.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/40per/test_04.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/40per/test_05.tsv",
        ]

        test_60OOC = [
            basePath + "data/ATIS/experiment/test/NoiseOOC/60per/test_01.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/60per/test_02.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/60per/test_03.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/60per/test_04.tsv",
            basePath + "data/ATIS/experiment/test/NoiseOOC/60per/test_05.tsv",
        ]

        test_clean = [basePath + "data/ATIS/experiment/test/clean/test.tsv"]

        return {
            "test_20OOC": test_20OOC,
            "test_40OOC": test_40OOC,
            "test_60OOC": test_60OOC,
            "test_25BG": test_25BG,
            "test_50BG": test_50BG,
            "test_75BG": test_75BG,
            "clean": test_clean,
        }

    if test == "SNIPS":

        test_50BG = [
            basePath + "data/SNIPS/experiments/test/NoiseBG/50per/test_01.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/50per/test_02.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/50per/test_03.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/50per/test_04.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/50per/test_05.tsv",
        ]

        test_25BG = [
            basePath + "data/SNIPS/experiments/test/NoiseBG/25per/test_01.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/25per/test_02.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/25per/test_03.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/25per/test_04.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/25per/test_05.tsv",
        ]

        test_75BG = [
            basePath + "data/SNIPS/experiments/test/NoiseBG/75per/test_01.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/75per/test_02.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/75per/test_03.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/75per/test_04.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseBG/75per/test_05.tsv",
        ]

        test_20OOC = [
            basePath + "data/SNIPS/experiments/test/NoiseOOC/20per/test_01.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/20per/test_02.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/20per/test_03.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/20per/test_04.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/20per/test_05.tsv",
        ]

        test_40OOC = [
            basePath + "data/SNIPS/experiments/test/NoiseOOC/40per/test_01.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/40per/test_02.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/40per/test_03.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/40per/test_04.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/40per/test_05.tsv",
        ]

        test_60OOC = [
            basePath + "data/SNIPS/experiments/test/NoiseOOC/60per/test_01.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/60per/test_02.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/60per/test_03.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/60per/test_04.tsv",
            basePath + "data/SNIPS/experiments/test/NoiseOOC/60per/test_05.tsv",
        ]

        test_clean = [basePath + "data/SNIPS/experiments/test/clean/test.tsv"]

        return {
            "test_20OOC": test_20OOC,
            "test_40OOC": test_40OOC,
            "test_60OOC": test_60OOC,
            "test_25BG": test_25BG,
            "test_50BG": test_50BG,
            "test_75BG": test_75BG,
            "clean": test_clean,
        }
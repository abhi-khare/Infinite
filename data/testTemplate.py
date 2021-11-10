def test_template(test, basePath):

    if test == "ATIS":

        test_clean = [basePath + "data/ATIS/experiments/clean/test/test.tsv"]

        test_BT_ru_1 = [basePath + "data/ATIS/experiments/AT/back_trans/test/ru/test_fbwmt19_ru_N3_F1_B1.tsv"]
        test_BT_ru_4 = [basePath + "data/ATIS/experiments/AT/back_trans/test/ru/test_fbwmt19_ru_N3_F4_B4.tsv"]
        test_BT_ru_16 = [basePath + "data/ATIS/experiments/AT/back_trans/test/ru/test_fbwmt19_ru_N3_F16_B16.tsv"]
        test_BT_ru_64 = [basePath + "data/ATIS/experiments/AT/back_trans/test/ru/test_fbwmt19_ru_N3_F64_B64.tsv"]

        test_BT_de_1 = [basePath + "data/ATIS/experiments/AT/back_trans/test/de/test_fbwmt19_de_N3_F1_B1.tsv"]
        test_BT_de_4 = [basePath + "data/ATIS/experiments/AT/back_trans/test/de/test_fbwmt19_de_N3_F4_B4.tsv"]
        test_BT_de_16 = [basePath + "data/ATIS/experiments/AT/back_trans/test/de/test_fbwmt19_de_N3_F16_B16.tsv"]
        test_BT_de_64 = [basePath + "data/ATIS/experiments/AT/back_trans/test/de/test_fbwmt19_de_N3_F64_B64.tsv"]

        return {
            "clean": test_clean,
            "test_BT_ru_1" : test_BT_ru_1,
            "test_BT_ru_4" : test_BT_ru_4,
            "test_BT_ru_16" : test_BT_ru_16,
            "test_BT_ru_64" : test_BT_ru_64,
            "test_BT_de_1" : test_BT_de_1,
            "test_BT_de_4" : test_BT_de_4,
            "test_BT_de_16" : test_BT_de_16, 
            "test_BT_de_64" : test_BT_de_64
        }

    if test == "SNIPS":

        test_clean = [basePath + "data/SNIPS/experiments/clean/test/test.tsv"]

        test_BT_ru_1 = [basePath + "data/SNIPS/experiments/back_trans/ru/test_fbwmt19_ru_N3_F1_B1.tsv"]
        test_BT_ru_4 = [basePath + "data/SNIPS/experiments/back_trans/ru/test_fbwmt19_ru_N3_F4_B4.tsv"]
        test_BT_ru_16 = [basePath + "data/SNIPS/experiments/back_trans/ru/test_fbwmt19_ru_N3_F16_B16.tsv"]
        test_BT_ru_64 = [basePath + "data/SNIPS/experiments/back_trans/ru/test_fbwmt19_ru_N3_F64_B64.tsv"]

        test_BT_de_1 = [basePath + "data/SNIPS/experiments/back_trans/de/test_fbwmt19_de_N3_F1_B1.tsv"]
        test_BT_de_4 = [basePath + "data/SNIPS/experiments/back_trans/de/test_fbwmt19_de_N3_F4_B4.tsv"]
        test_BT_de_16 = [basePath + "data/SNIPS/experiments/back_trans/de/test_fbwmt19_de_N3_F16_B16.tsv"]
        test_BT_de_64 = [basePath + "data/SNIPS/experiments/back_trans/de/test_fbwmt19_de_N3_F64_B64.tsv"]

        return {
            "clean": test_clean,
            "test_BT_ru_1" : test_BT_ru_1,
            "test_BT_ru_4" : test_BT_ru_4,
            "test_BT_ru_16" : test_BT_ru_16,
            "test_BT_ru_64" : test_BT_ru_64,
            "test_BT_de_1" : test_BT_de_1,
            "test_BT_de_4" : test_BT_de_4,
            "test_BT_de_16" : test_BT_de_16, 
            "test_BT_de_64" : test_BT_de_64
        }
    
    if test == "TOD":

        test_clean = [basePath + "data/TOD/experiments/clean/test/test.tsv"]

        test_BT_ru_1 = [basePath + "data/TOD/experiments/back_trans/ru/test_fbwmt19_ru_N3_F1_B1.tsv"]
        test_BT_ru_4 = [basePath + "data/TOD/experiments/back_trans/ru/test_fbwmt19_ru_N3_F4_B4.tsv"]
        test_BT_ru_16 = [basePath + "data/TOD/experiments/back_trans/ru/test_fbwmt19_ru_N3_F16_B16.tsv"]
        test_BT_ru_64 = [basePath + "data/TOD/experiments/back_trans/ru/test_fbwmt19_ru_N3_F64_B64.tsv"]

        test_BT_de_1 = [basePath + "data/TOD/experiments/back_trans/de/test_fbwmt19_de_N3_F1_B1.tsv"]
        test_BT_de_4 = [basePath + "data/TOD/experiments/back_trans/de/test_fbwmt19_de_N3_F4_B4.tsv"]
        test_BT_de_16 = [basePath + "data/TOD/experiments/back_trans/de/test_fbwmt19_de_N3_F16_B16.tsv"]
        test_BT_de_64 = [basePath + "data/TOD/experiments/back_trans/de/test_fbwmt19_de_N3_F64_B64.tsv"]

        return {
            "clean": test_clean,
            "test_BT_ru_1" : test_BT_ru_1,
            "test_BT_ru_4" : test_BT_ru_4,
            "test_BT_ru_16" : test_BT_ru_16,
            "test_BT_ru_64" : test_BT_ru_64,
            "test_BT_de_1" : test_BT_de_1,
            "test_BT_de_4" : test_BT_de_4,
            "test_BT_de_16" : test_BT_de_16, 
            "test_BT_de_64" : test_BT_de_64
        }
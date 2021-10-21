def test_template(test, basePath):

    if test == "ATIS":

        test_clean = [basePath + "data/ATIS/experiments/clean/test/test.tsv"]

        return {
            "clean": test_clean,
        }

    if test == "SNIPS":

        test_clean = [basePath + "data/SNIPS/experiments/clean/test/test.tsv"]

        return {
            "clean": test_clean,
        }
    
    if test == "TOD":

        test_clean = [basePath + "data/TOD/experiments/clean/test/test.tsv"]

        return {
            "clean": test_clean,
        }
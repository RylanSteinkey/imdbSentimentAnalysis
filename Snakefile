
rule all:
    input:
        "top_feats.npy"
rule data:
    output:
        "data.csv"
    shell:
        "python data_loader.py"

rule feat_builder:
    input:
        "data.csv"
    output:
        "words.pd"
    shell:
        "python feature_builder.py"

rule feat_select:
    input:
        "data.csv",
        "words.pd"
    output:
        "x_train.npy",
        "y_train.npy",
        "x_test.npy",
        "y_test.npy"
    shell:
        "python feature_builder.py"

rule models:
    input:
        "x_train.npy",
        "y_train.npy",
        "x_test.npy",
        "y_test.npy"
    output:
        "top_feats.npy"
    shell:
        "python models.py XGB"

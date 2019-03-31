
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
        "words.df"
    shell:
        "python feature_builder.py"

rule feat_select:
    input:
        "data.csv",
        "words.df"
    output:
        "data_sets/x_train.npy",
        "data_sets/y_train.npy",
        "data_sets/x_test.npy",
        "data_sets/y_test.npy"
    shell:
        "python feature_selection.py"

rule models:
    input:
        "data_sets/x_train.npy",
        "data_sets/y_train.npy",
        "data_sets/x_test.npy",
        "data_sets/y_test.npy"
    output:
        "top_feats.npy"
    shell:
        "python models.py XGB"

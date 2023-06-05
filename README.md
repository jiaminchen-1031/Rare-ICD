# Rare Codes Count: Mining Inter-code Relations for Long-tail Clinical Text Classification


## Requirement

```
conda env create -f environment.yaml
```


## Data

Arrange the data files in [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), and train-dev-test split files from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic), as below:

    Rare-ICD/
    |
    |--codes/
    |
    |--results/                              * Results and log files will be generated here                                  
    |
    |--mimicdata/                            * Put the downloaded MIMIC-III dataset here.
         |                                  
         |--NOTEEVENTS.csv                   * Collect from MIMIC-III.
         |
         |--DIAGNOSES_ICD.csv                * Collect from MIMIC-III.
         |
         |--PROCEDURES_ICD.csv               * Collect from MIMIC-III.
         |
         |--D_ICD_DIAGNOSES.csv              * Collect from MIMIC-III.
         |
         |--D_ICD_PROCEDURES.csv             * Collect from MIMIC-III.
         |
         |--ICD9_descriptions (Already given)
         |
         |--caml/                            * Train-dev-test split from caml-mimic.
         |    |
         |    |--train_50_hadm_ids.csv
         |    |
         |    |--dev_50_hadm_ids.csv
         |    |
         |    |--test_50_hadm_ids.csv
         |    |
         |    |--train_full_hadm_ids.csv
         |    |
         |    |--dev_full_hadm_ids.csv
         |    |
         |    |--test_full_hadm_ids.csv
         |
         |--generated_data/                   * The preprocessing codes will generate some files here.



## Adjacency Matrix 

1. Download the [adj files](https://drive.google.com/file/d/1LAluKX2kq-UvGrz_-tXbE_3WjhCOxVzG/view?usp=drive_link).

2. Put the adj.zip under /mimicdata.

3. unzip the file.

```
unzip adj.zip

rm -rf adj.zip
```

## Reproduction

1. Preprocess MIMIC-III data.

```
cd codes

python preprocess.py
```

2. Run the code.

Example:

```
CUDA_VISIBLE_DEVICES=1 python main.py \
--folder_path '../mimicdata/generated_data/' \
--make_sentence True \
--hidden_size 512 --name run_demo --data_setting 50 \
--rnn_layer 1 --max_len 4000 --learning_rate 0.001 \
--adj 'path-to-adj'
```

## Time Cost

We provide the training log examples of all our models, including the baseline as well, where the time for different stages are stamped for calculating the time cost for training and inference.

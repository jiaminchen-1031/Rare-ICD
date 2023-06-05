# PyTorch Implementation of [Multi-label Clinical Text Classification: Using Code Inner Relations to Guide Few-shot Learning]

## Requirement

```
conda env create -f environment.yaml
```


## Files Directory

Arrange the files in MIMIC-III from https://physionet.org/content/mimiciii/1.4/, as below:

    ICD_pred/
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
         |--caml/                            * train-dev-test split (already given) from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)
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
         |--generated_data/                       * The preprocessing codes will generate some files here.



## Reproduction

1. Preprocess MIMIC-III data.

```
cd ICD_pred

python preprocess.py
```

2. Run the code.

```
cd codes
```

Example:

```
CUDA_VISIBLE_DEVICES=1 python main.py --folder_path '../mimicdata/generated_data/' --make_sentence True --hidden_size 512 --name run_demo --data_setting 50 --rnn_layer 1 --max_len 4000 --learning_rate 0.001 --adj 'path-to-adj'
```

## Time Cost

We provide the training log examples of all our models, including the baseline as well, where the time for different stages are stamped for calculating the time cost for training and inference.
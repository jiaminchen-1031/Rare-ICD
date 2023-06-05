import logging
import random

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import constants
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import re
import string
from gensim.models import Word2Vec, FastText
import multiprocessing
from collections import defaultdict, Counter
import csv
import os
from tqdm import tqdm

tokenizer = RegexpTokenizer(r'\w+')


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def combine_diag_proc_codes(hadm_id_set, out_filename='ALL_CODES_filtered.csv'):
    logging.info("Started Preprocessing raw MIMIC-III data")
    diag_df = pd.read_csv(constants.DIAGNOSES_FILE_PATH, dtype={"ICD9_CODE": str})
    proc_df = pd.read_csv(constants.PORCEDURES_FILE_PATH, dtype={"ICD9_CODE": str})

    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].apply(lambda code: str(reformat(str(code), True)))
    proc_df['ICD9_CODE'] = proc_df['ICD9_CODE'].apply(lambda code: str(reformat(str(code), False)))
    codes_df = pd.concat([diag_df, proc_df], ignore_index=True)
    num_original_hadm_id = len(codes_df['HADM_ID'].unique())
    logging.info(f'Total unique HADM_ID (original): {num_original_hadm_id}')

    codes_df = codes_df[codes_df['HADM_ID'].isin(hadm_id_set)]
    codes_df.sort_values(['SUBJECT_ID', 'HADM_ID'], inplace=True)
    num_filtered_hadm_id = len(codes_df['HADM_ID'].unique())
    logging.info(f'Total unique HADM_ID (ALL_CODES_filtered): {num_filtered_hadm_id}')
    num_unique_codes = len(codes_df['ICD9_CODE'].unique())
    logging.info(f'Total unique ICD9_CODE (ALL_CODES_filtered): {num_unique_codes}')
    codes_df.to_csv(f'{constants.GENERATED_DIR}/{out_filename}', index=False,
               columns=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
               header=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])

    return out_filename


def contains_alphabetic(token):
    for c in token:
        if c.isalpha():
            return True
    return False


def clean_text(text):
    output = []
    length = 0

    for sent in sent_tokenize(text):
        tokens = [token.lower() for token in tokenizer.tokenize(sent) if contains_alphabetic(token)]
        length += len(tokens)

        sent = " ".join(tokens)

        if len(sent) > 0:
            output.append(sent)

    return "\n".join(output)


def write_discharge_summaries(out_filename='disch_full.csv'):
    logging.info("processing notes file")

    disch_df = pd.read_csv(constants.NOTEEVENTS_FILE_PATH)
    selected_categories = ['Discharge summary']
    disch_df = disch_df[disch_df['CATEGORY'].isin(selected_categories)]
    disch_df['TEXT'] = disch_df['TEXT'].apply(lambda text: clean_text(text))

    disch_df.sort_values(['SUBJECT_ID', 'HADM_ID'], inplace=True)
    logging.info(f'Total number of rows: {len(disch_df)}')
    hadm_id_set = set(disch_df['HADM_ID'])
    logging.info(f'Total number of unique HADM_ID (disch_full): {len(hadm_id_set)}')
    disch_df.to_csv(f'{constants.GENERATED_DIR}/{out_filename}', index=False,
               columns=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT'],
               header=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT'])

    return hadm_id_set, out_filename


def combine_notes_codes(disch_full_filename, filtered_codes_filename, out_filename='notes_labeled.csv'):
    logging.info('Merging discharge summary and ICD codes')
    disch_df = pd.read_csv(f'{constants.GENERATED_DIR}/{disch_full_filename}')
    disch_grouped = disch_df.groupby('HADM_ID')['TEXT'].apply(lambda texts: '\n'.join(texts))
    disch_df = pd.DataFrame(disch_grouped)
    disch_df.reset_index(inplace=True)

    codes_df = pd.read_csv(f'{constants.GENERATED_DIR}/{filtered_codes_filename}', dtype={"ICD9_CODE": str})
    codes_grouped = codes_df.groupby('HADM_ID')['ICD9_CODE'].apply(lambda codes: ';'.join(map(str, codes)))
    codes_df = pd.DataFrame(codes_grouped)
    codes_df.reset_index(inplace=True)

    merged_df = pd.merge(disch_df, codes_df, on='HADM_ID')
    merged_df.sort_values(['HADM_ID'], inplace=True)
    num_hadm_id = len(merged_df['HADM_ID'].unique())
    logging.info(f'Total number of unique HADM_ID (merged): {num_hadm_id}')

    merged_df.to_csv(f'{constants.GENERATED_DIR}/{out_filename}', index=False,
               columns=['HADM_ID', 'TEXT', 'ICD9_CODE'],
               header=['HADM_ID', 'TEXT', 'LABELS'])

    return out_filename


def split_data(labeled_notes_filename='notes_labeled.csv', is_full=True):
    labeled_notes_df = pd.read_csv(f'{constants.GENERATED_DIR}/{labeled_notes_filename}')
    counter = Counter()
    for labels in labeled_notes_df['LABELS'].values:
        for label in str(labels).split(';'):
            counter[label] += 1

    codes, freqs = map(list, zip(*counter.most_common()))
    code_freq_df = pd.DataFrame({'code': codes, 'freq': freqs})
    code_freq_filename = 'code_freq.csv'
    code_freq_df.to_csv(f'{constants.GENERATED_DIR}/{code_freq_filename}', index=False)

    if is_full:
        content = 'full'
        for split in ['train', 'dev', 'test']:
            split_hadm_id_df = pd.read_csv(f'{constants.CAML_DIR}/{split}_{content}_hadm_ids.csv', names=['HADM_ID'])
            split_hadm_ids = split_hadm_id_df['HADM_ID'].values.tolist()

            split_labeled_notes_df = labeled_notes_df[labeled_notes_df['HADM_ID'].isin(split_hadm_ids)]
            split_labeled_notes_df['LENGTH'] = split_labeled_notes_df['TEXT'].apply(lambda text: len(str(text).split()))
            split_labeled_notes_df = split_labeled_notes_df.sort_values(['LENGTH'], ascending=False)
            logging.info(f'Total rows in {split}_{content}.csv: {len(split_labeled_notes_df)}')
            split_labeled_notes_df.to_csv(f'{constants.GENERATED_DIR}/{split}_{content}.csv', index=False)
    else:
        content = 50
        top_codes = set(codes[:content])

        for split in ['train', 'dev', 'test']:
            split_hadm_id_df = pd.read_csv(f'{constants.CAML_DIR}/{split}_{content}_hadm_ids.csv', names=['HADM_ID'])
            split_hadm_ids = split_hadm_id_df['HADM_ID'].values.tolist()

            split_labeled_notes_df = labeled_notes_df[labeled_notes_df['HADM_ID'].isin(split_hadm_ids)]
            split_labeled_notes_df['LABELS'] = split_labeled_notes_df['LABELS'].apply(lambda codes: ';'.join(top_codes.intersection(set(codes.split(';')))))
            split_labeled_notes_df = split_labeled_notes_df[split_labeled_notes_df['LABELS'].str.len() > 0]

            split_labeled_notes_df['LENGTH'] = split_labeled_notes_df['TEXT'].apply(lambda text: len(str(text).split()))
            split_labeled_notes_df = split_labeled_notes_df.sort_values(['LENGTH'], ascending=False)
            logging.info(f'Total rows in {split}_{content}.csv: {len(split_labeled_notes_df)}')
            split_labeled_notes_df.to_csv(f'{constants.GENERATED_DIR}/{split}_{content}.csv', index=False)


def build_vocab(train_full_filename='train_full.csv', out_filename='vocab.csv'):
    train_df = pd.read_csv(f'{constants.GENERATED_DIR}/{train_full_filename}')
    desc_dt = load_code_desc()
    desc_series = pd.Series(list(desc_dt.values())).apply(lambda text: clean_text(text))

    full_text_series = train_df['TEXT'].append(desc_series, ignore_index=True)
    cv = CountVectorizer(min_df=1)
    cv.fit(full_text_series)

    out_file_path = f'{constants.GENERATED_DIR}/{out_filename}'
    with open(out_file_path, 'w') as fout:
        for word in cv.get_feature_names():
            fout.write(f'{word}\n')


def load_code_desc():
    desc_dict = defaultdict(str)
    with open(constants.DIAG_CODE_DESC_FILE_PATH, 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            desc_dict[reformat(code, True)] = desc
    with open(constants.PROC_CODE_DESC_FILE_PATH, 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            if code not in desc_dict.keys():
                desc_dict[reformat(code, False)] = desc
    with open(constants.ICD_DESC_FILE_PATH, 'r') as labelfile:
        for i,row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict


def embed_words(disch_full_filename='disch_full.csv', embed_size=128, out_filename='disch_full.w2v'):
    ######################共用一个embedding##################################
    disch_df = pd.read_csv(f'{constants.GENERATED_DIR}/{disch_full_filename}')
    sentences = []
    for text in disch_df['TEXT']:
        for sentence in text.split('\n'):
            sentences.append(sentence.split())
    desc_dt = load_code_desc()
    for desc in desc_dt.values():
        sentences.append(clean_text(desc).split())

    num_cores = int(multiprocessing.cpu_count()/2)
    min_count = 0
    window = 5
    num_negatives = 5
    logging.info('\n**********************************************\n')
    logging.info('Training CBOW embedding...')
    logging.info(f'Params: embed_size={embed_size}, workers={num_cores-1}, min_count={min_count}, window={window}, negative={num_negatives}')

    w2v_model = Word2Vec(min_count=min_count, window=window, size=embed_size, negative=num_negatives, workers=num_cores)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)
    w2v_model.init_sims(replace=True)
    w2v_model.save(f'{constants.GENERATED_DIR}/{out_filename}')

    logging.info('\n**********************************************\n')

    return out_filename


def map_vocab_to_embed(embed_filename='disch_full.w2v', out_filename='vocab.embed', laat=False):
    if not laat:
        model = Word2Vec.load(f'{constants.GENERATED_DIR}/{embed_filename}')
    else:
        model = Word2Vec.load(f'{constants.EMBED_LAAT_FILE_PATH}')

    wv = model.wv
    del model

    embed_size = len(wv.word_vec(wv.index2word[0]))
    word_to_idx = {}
    with open(f'{constants.GENERATED_DIR}/{out_filename}', 'w') as fout:
        pad_embed = np.zeros(embed_size)
        unk_embed = np.random.randn(embed_size)
        unk_embed_normalized = unk_embed / float(np.linalg.norm(unk_embed) + 1e-6)
        fout.write(constants.PAD_SYMBOL + ' ' + np.array2string(pad_embed, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
        fout.write(constants.UNK_SYMBOL + ' ' + np.array2string(unk_embed_normalized, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
        word_to_idx[constants.PAD_SYMBOL] = 0
        word_to_idx[constants.UNK_SYMBOL] = 1

        for item in wv.vocab:
            word_embed = wv.word_vec(item)
            fout.write(item + ' ' + np.array2string(word_embed, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
            word_to_idx[item] = len(word_to_idx)

    logging.info(f'Size of training vocabulary (including PAD, UNK): {len(word_to_idx)}')
    return word_to_idx


def vectorize_code_desc(word_to_idx, out_filename='code_desc_vectors.csv'):
    desc_dict = load_code_desc()
    with open(f'{constants.GENERATED_DIR}/{out_filename}', 'w') as fout:
        w = csv.writer(fout, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for code, desc in desc_dict.items():
            tokens = clean_text(desc).split()
            inds = [word_to_idx[t] if t in word_to_idx.keys() else word_to_idx[constants.UNK_SYMBOL] for t in tokens]
            w.writerow([code] + [str(i) for i in inds])


if __name__ == '__main__':
    if not os.path.exists('../results'):
        os.makedirs('../results')
    args = constants.get_args()
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='../results/preprocess.log', filemode='w', format=FORMAT, level=logging.INFO)
    hadm_id_set, disch_full_filename = write_discharge_summaries()
    filtered_codes_filename = combine_diag_proc_codes(hadm_id_set)
    labeled_notes_filename = combine_notes_codes(disch_full_filename, filtered_codes_filename)
    split_data()
    split_data(is_full=False)
    build_vocab()
    embed_filename = embed_words(embed_size=128)
    word_to_idx = map_vocab_to_embed()
    vectorize_code_desc(word_to_idx)


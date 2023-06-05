import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from utils import *
from constants import *
import pandas as pd
import datetime
import sentencepiece as spm
from scipy.stats import bernoulli
import random


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


def load_adj_matrix(data_setting, path):
    adj_matrix = torch.load(path, encoding='latin1')
    if data_setting == TOP50:
        adj = adj_matrix[:50, :50].float()
        adj = adj.numpy()
        adj += np.eye(adj.shape[0])
        degree = np.array(adj.sum(1))
        degree = np.diag(np.power(degree, -0.5))
        out = degree.dot(adj).dot(degree)
        out = torch.as_tensor(out, dtype=torch.float32)
        return out

    adj = adj_matrix.numpy()
    np.add(adj, np.eye(adj.shape[0]), out=adj, casting="unsafe")
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    out = degree.dot(adj).dot(degree)
    out = torch.as_tensor(out, dtype=torch.float32)
    return out


def remove_stopwords(text):
    stpwords = set([stopword for stopword in stopwords.words('english')])
    stpwords.update({'admission', 'birth', 'date', 'discharge', 'service', 'sex'})
    tokens = text.strip().split()
    tokens = [token for token in tokens if token not in stpwords]
    return ' '.join(tokens)


def load_dataset(data_setting, batch_size, split):
    data = pd.read_csv(f'{FILE_DIR}/{split}_{data_setting}.csv', dtype={'LENGTH': int})
    len_stat = data['LENGTH'].describe()
    logging.info(f'{split} set length stats:\n{len_stat}')

    code_df = pd.read_csv(f'{CODE_FREQ_PATH}', dtype={'code': str})
    if data_setting == FULL:
        all_codes = ';'.join(map(str, code_df['code'].values.tolist()))
        data = data.append({'HADM_ID': -1, 'TEXT': 'remove', 'LABELS': all_codes, 'length': 6},
                           ignore_index=True)

    code_class = code_df['code'].values.tolist()
    code_class_50 = code_df['code'].values.tolist()[:50]
    if data_setting == FULL:
        mlb = MultiLabelBinarizer(classes=code_class)
    else:
        mlb = MultiLabelBinarizer(classes=code_class_50)
    data['LABELS'] = data['LABELS'].apply(lambda x: str(x).split(';'))
    code_counts = list(data['LABELS'].str.len())
    avg_code_counts = sum(code_counts) / len(code_counts)
    logging.info(f'In {split} set, average code counts per discharge summary: {avg_code_counts}')
    mlb.fit(data['LABELS'])
    temp = mlb.transform(data['LABELS'])
    if mlb.classes_[-1] == 'nan':
        mlb.classes_ = mlb.classes_[:-1]
    logging.info(f'Final number of labels/codes: {len(mlb.classes_)}')

    for i, x in enumerate(mlb.classes_):
        data[x] = temp[:, i]
    data.drop(['LABELS', 'LENGTH'], axis=1, inplace=True)

    if data_setting == FULL:
        data = data[:-1]

    code_list = list(mlb.classes_)
    label_freq = list(data[code_list].sum(axis=0))
    hadm_ids = data['HADM_ID'].values.tolist()
    texts = data['TEXT'].values.tolist()
    labels = data[code_list].values.tolist()
    item_count = (len(texts) // batch_size) * batch_size
    logging.info(f'{split} set true item count: {item_count}\n\n')
    return {'hadm_ids': hadm_ids[:item_count],
            'texts': texts[:item_count],
            'targets': labels[:item_count],
            'labels': code_list,
            'label_freq': label_freq}


def get_all_codes(train_path, dev_path, test_path):
    all_codes = set()
    splits_path = {'train': train_path, 'dev': dev_path, 'test': test_path}
    for split, file_path in splits_path.items():
        split_df = pd.read_csv(file_path, dtype={'HADM_ID': str})
        split_codes = set()
        for codes in split_df['LABELS'].values:
            for code in str(codes).split(';'):
                split_codes.add(code)

        logging.info(f'{split} set has {len(split_codes)} unique codes')
        all_codes.update(split_codes)

    logging.info(f'In total, there are {len(all_codes)} unique codes')
    return list(all_codes)


def replace_word_with_semtypes(texts, word2sem, p=0.015):
    texts_replaced = []
    for text in texts:
        sentences_replaced = []
        sentences = text.split('\n')
        for sentence in sentences:
            tokens = sentence.split()
            for id, token in enumerate(tokens):
                if token in word2sem.keys():
                    a = bernoulli.rvs(p)
                    if a:
                        codes = word2sem[token]
                        code = np.random.choice(codes)
                        tokens[id] = code
            sentences_replaced.append(' '.join(tokens))
        random.shuffle(sentences_replaced)
        texts_replaced.append('\n'.join(sentences_replaced))
        
    return texts_replaced


def load_datasets(data_setting, batch_size, word2sem, nfold):
    train_raw = load_dataset(data_setting, batch_size, split='train')
    dev_raw = load_dataset(data_setting, batch_size, split='dev')
    test_raw = load_dataset(data_setting, batch_size, split='test')

    texts = train_raw['texts'].copy()
    hadm_ids = train_raw['hadm_ids'].copy()
    targets = train_raw['targets'].copy()

    # for _ in range(nfold):
    #     train_raw['texts'].extend(replace_word_with_semtypes(texts, word2sem))
    #     train_raw['hadm_ids'].extend(hadm_ids)
    #     train_raw['targets'].extend(targets)

    if train_raw['labels'] != dev_raw['labels'] or dev_raw['labels'] != test_raw['labels']:
        raise ValueError(f"Train dev test labels don't match!")

    return train_raw, dev_raw, test_raw


def load_embedding_weights():
    W = []
    # PAD and UNK already in embed file

    with open(EMBED_FILE_PATH) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            # vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
    logging.info(f'Total token count (including PAD, UNK) of full preprocessed discharge summaries: {len(W)}')
    weights = torch.as_tensor(W, dtype=torch.float)
    return weights


def get_proc_diag_code(data_setting):
    diag_df = pd.read_csv(DIAGNOSES_FILE_PATH,  dtype={"ICD9_CODE": str})

    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].apply(lambda code: str(reformat(str(code), True)))

    diag_list = diag_df['ICD9_CODE'].unique().tolist()

    code_df = pd.read_csv(f'{CODE_FREQ_PATH}', dtype={'code': str})
    code_class = code_df['code'].values.tolist()

    for id, j in enumerate(code_class):
        if j in diag_list:
            code_class[id] = True
        else:
            code_class[id] = False
    print('code_class_50', code_class[:50])

    embedding=[]
    with open(PROC_DIAG_EMBEDDING) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            embedding.append(vec)

    if data_setting == TOP50:
        return code_class[:50], embedding[0], embedding[1]

    return code_class, embedding[0], embedding[1]


def load_umls_embedding_weights():
    W = []
    # PAD and UNK already in embed file

    with open(UMLS_EMBED_FILE_PATH) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            # vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
    logging.info(f'Finish loading all umls embedding ...')
    weights = torch.as_tensor(W, dtype=torch.float)
    return weights


def load_label_embedding(labels, pad_index, embed_size):
    code_desc = []
    desc_dt = {}
    max_desc_len = 0
    with open(f'{CODE_DESC_VECTOR_PATH}', 'r') as fin:
        for line in fin:
            if line != '\n':
                items = line.strip().split()
                code = items[0]
                if code in labels:
                    desc_dt[code] = list(map(int, items[1:]))
                    max_desc_len = max(max_desc_len, len(desc_dt[code]))
    for code in labels:
        pad_len = max_desc_len - len(desc_dt[code])
        code_desc.append(desc_dt[code] + [pad_index] * pad_len)

    code_desc = torch.as_tensor(code_desc, dtype=torch.long)

    label2, label3 = classify_level_1_level_2(labels)

    label2_array = np.array(label2).flatten()
    label3_array = np.array(label3).flatten()

    label2_unique = np.unique(label2_array)
    label3_unique = np.unique(label3_array)

    nb2 = len(label2_unique)
    nb1 = len(label3_unique)
    nb0 = len(labels)

    label_index_2 = torch.zeros((nb2, nb0, embed_size))
    label_index_1 = torch.zeros((nb1, nb0, embed_size))

    for i in range(nb2):
        label_index_2[i, np.where(label2_array == label2_unique[i]), :] = True

    for i in range(nb1):
        label_index_1[i, np.where(label3_array == label3_unique[i]), :] = True

    return code_desc, label_index_1, label_index_2


def index_text(data, indexer, max_len, bpe, split):
    if bpe:
        tokenizer = spm.SentencePieceProcessor(model_file='bpe3.model')
    data_indexed = []
    lens = []
    # count = 0
    oov_word_frac = []
    print('if bpe', bpe)
    for text in data:
        num_oov_words = 0
        text_indexed = [indexer.index_of(PAD_SYMBOL)] * max_len

        if bpe:
            text_s = text.split()
            tokens = []
            for word in text_s:
                tokens.extend(tokenizer.encode(word, out_type=str))
          ########word-level token######################
        else:
            tokens = text.split()
            #############################################

        text_len = max_len if len(tokens) > max_len else len(tokens)
        lens.append(text_len)
        for i in range(text_len):

            if indexer.index_of(tokens[i]) >= 0:
                text_indexed[i] = indexer.index_of(tokens[i])
            else:
                num_oov_words += 1
                text_indexed[i] = indexer.index_of(UNK_SYMBOL)

        oov_word_frac.append(num_oov_words / text_len)
        data_indexed.append(text_indexed)

    logging.info(
        f'{split} dataset has on average {sum(oov_word_frac) / len(oov_word_frac)} oov words per discharge summary')

    return data_indexed, lens


class ICD_Dataset(Dataset):
    def __init__(self, hadm_ids, texts, lens, labels):
        self.hadm_ids = hadm_ids
        self.texts = texts
        self.lens = lens
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def get_code_count(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        hadm_id = torch.tensor(self.hadm_ids[index])
        text = torch.tensor(self.texts[index], dtype=torch.long)
        length = torch.tensor(self.lens[index], dtype=torch.long)
        codes = torch.tensor(self.labels[index], dtype=torch.float)
        return {'hadm_id': hadm_id, 'text': text, 'length': length, 'codes': codes}


def prepare_datasets(data_setting, batch_size, max_len, bpe=None, nfold=None):

    # word2sem = dict()
    # df_text = pd.read_csv('./processed_discharge.csv', sep=';')
    # for i in range(len(df_text)):
    #     word2sem[df_text['Words'][i]] = df_text['Semtypes'][i].split(',')
    word2sem = None

    train_data, dev_data, test_data = load_datasets(data_setting, batch_size, word2sem, nfold)

    input_indexer = Indexer()
    input_indexer.add_and_get_index(PAD_SYMBOL)
    input_indexer.add_and_get_index(UNK_SYMBOL)
    with open(EMBED_FILE_PATH, 'r') as fin, open('vocab_word.csv', 'w') as fout:
        for line in fin:
            line = line.rstrip().split()
            word = line[0]
            input_indexer.add_and_get_index(word)
            fout.write(word + '\n')

    logging.info(f'Size of training vocabulary including PAD, UNK: {len(input_indexer)}')

    train_text_indexed, train_lens = index_text(train_data['texts'], input_indexer, max_len, bpe, split='train')
    dev_text_indexed, dev_lens = index_text(dev_data['texts'], input_indexer, max_len, bpe, split='dev')
    test_text_indexed, test_lens = index_text(test_data['texts'], input_indexer, max_len, bpe, split='test')

    # num = torch.sum(torch.tensor(train_data['targets']), axis=1)
    # print(num)
    # print ('Max_number =', max(num))

    train_set = ICD_Dataset(train_data['hadm_ids'], train_text_indexed, train_lens, train_data['targets'])
    dev_set = ICD_Dataset(dev_data['hadm_ids'], dev_text_indexed, dev_lens, dev_data['targets'])
    test_set = ICD_Dataset(test_data['hadm_ids'], test_text_indexed, test_lens, test_data['targets'])

    return train_set, dev_set, test_set, train_data['labels'], train_data['label_freq'], input_indexer


def classify_level_1_level_2(labels):
    class001_139 = []
    class140_239 = []
    class240_279 = []
    class280_289 = []
    class290_319 = []
    class320_359 = []
    class360_389 = []
    class390_459 = []
    class460_519 = []
    class520_579 = []
    class580_629 = []
    class630_679 = []
    class680_709 = []
    class710_739 = []
    class740_759 = []
    class760_779 = []
    class780_799 = []
    class800_999 = []
    classV = []
    classE = []

    class00 = []
    class01_05 = []
    class06_07 = []
    class08_16 = []
    class17 = []
    class18_20 = []
    class21_29 = []
    class30_34 = []
    class35_39 = []
    class40_41 = []
    class42_54 = []
    class55_59 = []
    class60_64 = []
    class65_71 = []
    class72_75 = []
    class76_84 = []
    class85_86 = []
    class87_99 = []

    labels3 = np.zeros(shape = np.array(labels).shape)
    labels3 = labels3.tolist()

    for index, each in enumerate(labels):
        class1 = each.split(".")[0]
        if len(class1) >= 3 :
            if class1[0] == 'V':
                classV.append(index)
            elif class1[0] == 'E':
                classE.append(index)
            else:
                class1_int = int(class1)
                if class1_int <= 139:
                    class001_139.append(index)
                elif class1_int <= 239:
                    class140_239.append(index)
                elif class1_int <= 279:
                    class240_279.append(index)
                elif class1_int <= 289:
                    class280_289.append(index)
                elif class1_int <= 319:
                    class290_319.append(index)
                elif class1_int <= 359:
                    class320_359.append(index)
                elif class1_int <= 389:
                    class360_389.append(index)
                elif class1_int <= 459:
                    class390_459.append(index)
                elif class1_int <= 519:
                    class460_519.append(index)
                elif class1_int <= 579:
                    class520_579.append(index)
                elif class1_int <= 629:
                    class580_629.append(index)
                elif class1_int <= 679:
                    class630_679.append(index)
                elif class1_int <= 709:
                    class680_709.append(index)
                elif class1_int <= 739:
                    class710_739.append(index)
                elif class1_int <= 759:
                    class740_759.append(index)
                elif class1_int <= 779:
                    class760_779.append(index)
                elif class1_int <= 799:
                    class780_799.append(index)
                else:
                    class800_999.append(index)
        else:
            class1_int = int(class1)
            labels3[index] = class1
            if class1_int <= 0:
                class00.append(index)
            elif class1_int <= 5:
                class01_05.append(index)
            elif class1_int <= 7:
                class06_07.append(index)
            elif class1_int <= 16:
                class08_16.append(index)
            elif class1_int <= 17:
                class17.append(index)
            elif class1_int <= 20:
                class18_20.append(index)
            elif class1_int <= 29:
                class21_29.append(index)
            elif class1_int <= 34:
                class30_34.append(index)
            elif class1_int <= 39:
                class35_39.append(index)
            elif class1_int <= 41:
                class40_41.append(index)
            elif class1_int <= 54:
                class42_54.append(index)
            elif class1_int <= 59:
                class55_59.append(index)
            elif class1_int <= 64:
                class60_64.append(index)
            elif class1_int <= 71:
                class65_71.append(index)
            elif class1_int <= 75:
                class72_75.append(index)
            elif class1_int <= 84:
                class76_84.append(index)
            elif class1_int <= 86:
                class85_86.append(index)
            else:
                class87_99.append(index)

    labels2 = np.zeros(shape = np.array(labels).shape)
    labels2 = labels2.tolist()

    for id in class001_139:
        labels2[id] = '001'
    for id in class140_239:
        labels2[id] = '140'
    for id in class240_279:
        labels2[id] = '240'
    for id in class280_289:
        labels2[id] = '280'
    for id in class290_319:
        labels2[id] = '290'
    for id in class320_359:
        labels2[id] = '320'
    for id in class360_389:
        labels2[id] = '360'
    for id in class390_459:
        labels2[id] = '390'
    for id in class460_519:
        labels2[id] = '460'
    for id in class520_579:
        labels2[id] = '520'
    for id in class580_629:
        labels2[id] = '580'
    for id in class630_679:
        labels2[id] = '630'
    for id in class680_709:
        labels2[id] = '680'
    for id in class710_739:
        labels2[id] = '710'
    for id in class740_759:
        labels2[id] = '740'
    for id in class760_779:
        labels2[id] = '760'
    for id in class780_799:
        labels2[id] = '780'
    for id in class800_999:
        labels2[id] = '800'
    for id in class00:
        labels2[id] = '00'
    for id in class01_05:
        labels2[id] = '01'
    for id in class06_07:
        labels2[id] = '06'
    for id in class08_16:
        labels2[id] = '08'
    for id in class17:
        labels2[id] = '17'
    for id in class18_20:
        labels2[id] = '18'
    for id in class21_29:
        labels2[id] = '21'
    for id in class30_34:
        labels2[id] = '30'
    for id in class35_39:
        labels2[id] = '35'
    for id in class40_41:
        labels2[id] = '40'
    for id in class42_54:
        labels2[id] = '42'
    for id in class55_59:
        labels2[id] = '55'
    for id in class60_64:
        labels2[id] = '60'
    for id in class65_71:
        labels2[id] = '65'
    for id in class72_75:
        labels2[id] = '72'
    for id in class76_84:
        labels2[id] = '76'
    for id in class85_86:
        labels2[id] = '85'
    for id in class87_99:
        labels2[id] = '87'
    for id in classV:
        labels2[id] = 'V'
    for id in classE:
        labels2[id] = 'E'

    for index2 in class001_139:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([1, 10, 20, 30, 42, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 137, 140], code1)
        labels3[index2] = '0'*(3-len(str(label_i))) + str(label_i)

    for index2 in class140_239:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([140, 150, 160, 170, 176, 179, 190, 200, 209, 210, 230, 235, 239, 240], code1)
        labels3[index2] = str(label_i)

    for index2 in class240_279:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([240, 249, 260, 270, 280], code1)
        labels3[index2] = str(label_i)

    for index2 in class280_289:
        labels3[index2] = '280'

    for index2 in class290_319:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([290, 300, 295, 317, 320], code1)
        labels3[index2] = str(label_i)

    for index2 in class320_359:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([320, 330, 338, 339, 340, 350, 360], code1)
        labels3[index2] = str(label_i)

    for index2 in class360_389:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([360, 380, 390], code1)
        labels3[index2] = str(label_i)

    for index2 in class390_459:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([390, 393, 401, 410, 415, 420, 430, 440, 451, 460], code1)
        labels3[index2] = str(label_i)

    for index2 in class460_519:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([460, 470, 480, 490, 500, 510, 520], code1)
        labels3[index2] = str(label_i)

    for index2 in class520_579:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([520, 530, 540, 550, 555, 560, 570, 580], code1)
        labels3[index2] = str(label_i)

    for index2 in class580_629:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([580, 590, 600, 610, 614, 617, 630], code1)
        labels3[index2] = str(label_i)

    for index2 in class630_679:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([630, 640, 650, 660, 670, 678, 680], code1)
        labels3[index2] = str(label_i)

    for index2 in class680_709:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([680, 690, 700, 710], code1)
        labels3[index2] = str(label_i)

    for index2 in class710_739:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([710, 720, 725, 730, 740], code1)
        labels3[index2] = str(label_i)

    for index2 in class740_759:
        labels3[index2] = '740'

    for index2 in class760_779:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([760, 764, 780], code1)
        labels3[index2] = str(label_i)

    for index2 in class780_799:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([780, 790, 797], code1)
        labels3[index2] = str(label_i)

    for index2 in class800_999:
        code = labels[index2]
        code1 = int(code.split(".")[0])
        label_i = find_code_position([800, 805, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 905, 910, 920, 925,
                                      930, 940, 950, 958, 960, 980, 990, 996, 1000], code1)
        labels3[index2] = str(label_i)

    for index2 in classE:
        code = labels[index2]
        code1 = int(code[1:].split(".")[0])
        label_i = find_code_position([0, 1, 800, 810, 820, 826, 830, 840, 846, 850, 860, 870, 878, 880, 890, 900, 910,
                                      916, 929, 930, 950, 960, 970, 980, 990, 1000], code1)
        labels3[index2] = 'E'+ '0'*(3-len(str(label_i))) + str(label_i)

    for index2 in classV:
        code = labels[index2]
        code1 = int(code[1:].split(".")[0])
        label_i = find_code_position([1, 10, 20, 30, 50, 60, 70, 83, 85, 86, 87, 88, 89, 90, 91, 100], code1)
        labels3[index2] = 'V' + '0'*(2-len(str(label_i))) + str(label_i)

    return labels2, labels3


def find_code_position(nodes, code):
    for i in range(len(nodes)-1):
        if nodes[i] <= code < nodes[i+1]:
            return nodes[i]
        else:
            continue
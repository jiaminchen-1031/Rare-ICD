from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import datetime

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


class InputData:
    def __init__(self, sentences, triplets_file_path, umls_dic_path, min_count):
        self.sentences = sentences
        self.df_triplets = pd.read_csv(triplets_file_path, sep=';')
        self.triplets_id = []
        self.entity_count = 0
        self.relation_count = 0
        self.df_umls = pd.read_csv(umls_dic_path, sep=';')
        self.min_count = min_count
        self.wordId_frequency_dict = dict() 
        self.word_count = 0 
        self.word_count_sum = 0 
        self.sentence_count = 0
        self.id2word_dict = dict() 
        self.vocab2id_dic = dict()
        self.umls_dic = dict()
        self._init_dict() 
        self.sample_table = []
        self._init_sample_table()

        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
        print('Relation Count is:', self.relation_count)
        print('Entity Count is', self.entity_count)
        print('Vocab Count is', len(self.vocab2id_dic))

    def _init_dict(self):
        word_freq = dict()
        for line in self.sentences:
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for word in line:
                try:
                    word_freq[word] += 1
                except:
                    word_freq[word] = 1
        word_id = 0
        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.vocab2id_dic[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = word_id

        ent_id = 0
        for ent in self.df_triplets['E1'].unique().tolist():
            if ent in self.vocab2id_dic.keys():
                continue
            else:
                self.vocab2id_dic[ent] = word_id
                word_id += 1
                ent_id += 1

        for ent in self.df_triplets['E2'].unique().tolist():
            if ent in self.vocab2id_dic.keys():
                continue
            else:
                self.vocab2id_dic[ent] = word_id
                ent_id += 1
                word_id += 1
        self.entity_count = ent_id

        rel_id = 0
        for rel in self.df_triplets['R'].unique().tolist():
            if rel in self.vocab2id_dic.keys():
                continue
            else:
                self.vocab2id_dic[rel] = rel_id
                rel_id += 1
                word_id += 1
        self.relation_count = rel_id

        for id in range(len(self.df_umls)):
            word = self.df_umls['Words'][id]
            sems = self.df_umls['Semtypes'][id].split(',')
            if word in self.umls_dic.keys():
                continue
            else:
                self.umls_dic[word] = sems

        for i in range(len(self.df_triplets)):
            e1_id = self.vocab2id_dic[self.df_triplets['E1'][i]]
            e2_id = self.vocab2id_dic[self.df_triplets['E2'][i]]
            r_id = self.vocab2id_dic[self.df_triplets['R'][i]]
            self.triplets_id.append([e1_id, r_id, e2_id])
            
        self.triplets_id = torch.LongTensor(self.triplets_id)

    def _init_sample_table(self):
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.wordId_frequency_dict.values())) ** 0.75  
        word_pow_sum = sum(pow_frequency) 
        ratio_array = pow_frequency / word_pow_sum 
        word_count_list = np.round(ratio_array * sample_table_size)
        for word_index, word_freq in enumerate(word_count_list):
            self.sample_table += [word_index] * int(word_freq) 
        self.sample_table = np.array(self.sample_table)

    def get_positive_pairs(self, window_size):
        result_pairs = []
        for sentence in self.sentences:
            if sentence is None or sentence == '':
                continue
            wordId_list = []  
            for word in sentence:
                try:
                    word_id = self.vocab2id_dic[word]
                    wordId_list.append(word_id)
                except:
                    continue
        
            for i, wordId_w in enumerate(wordId_list):
                context_ids = [len(self.vocab2id_dic)]*(2*window_size+1)
                umls_ent_ids = [len(self.vocab2id_dic)]*3
                if self.id2word_dict[wordId_w] in self.umls_dic.keys():
                    for id, k in enumerate(self.umls_dic[self.id2word_dict[wordId_w]]):
                        umls_ent_ids[id] = self.vocab2id_dic[k]

                for j, wordId_u in enumerate(wordId_list[max(i - window_size, 0):i + window_size + 1]):
                    assert wordId_w < self.word_count
                    assert wordId_u < self.word_count
                    if i == j:  
                        continue
                    elif max(0, i - window_size + 1) <= j <= min(len(wordId_list), i + window_size - 1):
                        context_ids[j] = wordId_u
                if context_ids == [len(self.vocab2id_dic)]*(2*window_size+1):
                    continue
                result_pairs.append((umls_ent_ids, context_ids, wordId_w))

        print('result_pairs', len(result_pairs))

        return result_pairs

    def get_negative_sampling(self, positive_pairs, neg_count):
        neg_u = np.random.choice(self.sample_table, size=(len(positive_pairs), neg_count)).tolist()
        return neg_u


class Input_Dataset(Dataset):
    def __init__(self, sentences, triplets_file_path, umls_dic_path, min_count, window_size, neg_count):
        self.data = InputData(sentences, triplets_file_path, umls_dic_path, min_count)
        self.pos_pairs = self.data.get_positive_pairs(window_size)
        self.neg_pairs = self.data.get_negative_sampling(self.pos_pairs, neg_count)

        self.pos_u = [pair[1] for pair in self.pos_pairs]
        self.pos_w = [int(pair[2]) for pair in self.pos_pairs]
        self.umls_t = [pair[0] for pair in self.pos_pairs]
        self.triplets = self.data.triplets_id
        self.neg_w = self.neg_pairs


    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):

        pos_u = torch.tensor(self.pos_u[index], dtype=torch.long)
        pos_w = torch.tensor(self.pos_w[index], dtype=torch.long)
        umls_t = torch.tensor(self.umls_t[index], dtype=torch.long)
        neg_w = torch.tensor(self.neg_w[index], dtype=torch.long)
        triplets = torch.tensor(self.triplets, dtype=torch.long)
        
        return {'pos_u': pos_u, 'pos_w': pos_w, 'umls_t': umls_t, 'neg_w': neg_w, 'triplets': triplets}


class CBOWModel(torch.nn.Module):
    """ Context words as input, returns possiblity distribution
        prediction of center word (target).
    """
    def __init__(self, device, ent_size, rel_size, vocabulary_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.device = device
        self.embeddings = torch.nn.Embedding((vocabulary_size+ent_size+rel_size+1), embedding_dim, padding_idx=(vocabulary_size+ent_size+rel_size))
        self.linear1 = torch.nn.Linear(embedding_dim, (vocabulary_size+ent_size+rel_size+1))
        self.layernorm = torch.nn.LayerNorm(embedding_dim)
        self.embed_dim = embedding_dim
        self.vocab_nb = vocabulary_size+ent_size+rel_size+1
        self.word_count =  vocabulary_size

    def initialize(self, pretrained_weight):
        self.embeddings.weight.data[:self.word_count] = pretrained_weight
        initrange = 0.5 / self.embed_dim
        self.embeddings.weight.data[self.word_count:-1].uniform_(-initrange, initrange)
        
    
    def forward(self, word, entities, contexts, negative, triplets): #[bs,1] [bs, 1] [bs, 1] [bs, neg] [nb_training, 3]
        # input

        e_embeds = self.embeddings(entities)
        c_embeds = self.embeddings(contexts)
        embeds = torch.sum(e_embeds, axis=1) + torch.sum(c_embeds, axis=1)

        # output
        out = self.layernorm(embeds)
        out = self.linear1(out)
        mask = torch.ones_like(out, dtype=torch.bool)
        mask = mask.scatter(1, negative, False)
        mask = mask.scatter(1, contexts, False)
        mask = mask.scatter(1, word.unsqueeze(1), False)
        masked_output = out.masked_fill(mask, -np.inf)

        out = F.log_softmax(masked_output, dim=1)
        criterion = nn.NLLLoss(reduction='mean')
        loss_cbow = criterion(out, word)

        r_embed = self.embeddings(triplets[:, :, 1])
        e1_embed = self.embeddings(triplets[:, :, 0])
        e2_embed = self.embeddings(triplets[:, :, 2])
        
        score_kgs_1 = torch.nn.functional.cosine_similarity((r_embed+e1_embed),e2_embed, dim=2)
        loss_kgs =  -F.logsigmoid(torch.mean(score_kgs_1))
        

#         score_kgs_1 = torch.mul((e1_embed + r_embed), e2_embed).squeeze()
#         score_kgs_2 = torch.sum(score_kgs_1, dim=2)
#         score_kgs_3 = torch.sum(score_kgs_2, dim=1)[0]
#         loss_kgs = -F.logsigmoid(score_kgs_3)

        
        return loss_cbow, loss_kgs

    def get_embeddings(self):
        return self.embeddings.weight.data.detach().cpu().numpy()


class Word2VecUMLS:
    def __init__(self, sentences, triplets_file_path, umls_dic_path, min_count, output_file_name, device, embed_size, window_size, batch_size, neg_count):
        self.output_file_name = output_file_name
        self.data = Input_Dataset(sentences, triplets_file_path, umls_dic_path, min_count, window_size, neg_count)
        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model = CBOWModel(device, self.data.data.entity_count, self.data.data.relation_count, self.data.data.word_count, embed_size).to(device)
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = device
        self.batch_size = batch_size
        self.embed_size = embed_size
    
    
    def train(self, epochs):

        writer = SummaryWriter()

        print("CBOW Training......")

        for epoch in range(epochs):
            loss_epoch = 0
            loss_epoch_cbow = 0
            loss_epoch_kgs = 0
            process_bar = tqdm(self.loader)
            for i, batch in enumerate(process_bar):
                
                # print('0')
                # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                pos_u = batch['pos_u']
                pos_w = batch['pos_w']
                umls_t = batch['umls_t']
                neg_w = batch['neg_w']
                triplet = batch['triplets']
                
                # print('1')
                # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                pos_u = pos_u.to(self.device)
                pos_w = pos_w.to(self.device)
                umls_t =umls_t.to(self.device)
                neg_w = neg_w.to(self.device)
                triplet = triplet.to(self.device)
                
                # print('2')
                # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                self.optimizer.zero_grad()
                loss_cbow, loss_kgs = self.model.forward(pos_w, umls_t, pos_u, neg_w, triplet)
                
                # print('3')
                # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                loss = loss_cbow + loss_kgs
                loss.backward()
                
                # print('4')
                # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.optimizer.step()
                
                # print('5')
                # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                writer.add_scalar('loss', loss, (i + epoch*self.batch_size))
                writer.add_scalar('loss_cbow', loss_cbow,  (i + epoch*self.batch_size))
                writer.add_scalar('loss_kgs', loss_kgs, (i + epoch * self.batch_size))
                
                loss_epoch += loss
                loss_epoch_cbow += loss_cbow
                loss_epoch_kgs += loss_kgs
                
            
            writer.add_scalar('loss_epoch', loss_epoch, epoch)
            writer.add_scalar('loss_epoch_cbow', loss_epoch_cbow,  epoch)
            writer.add_scalar('loss_epoch_kgs', loss_epoch_kgs,  epoch)
            print('EPOCH', epoch)
            print('LOSS', (loss_epoch, loss_epoch_cbow, loss_epoch_kgs))

        embedding = self.model.get_embeddings()
        print('pad', embedding[-1] == np.zeros(self.embed_size))
        
        return embedding

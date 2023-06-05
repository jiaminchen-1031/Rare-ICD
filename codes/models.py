import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input, adj):
        output = torch.mm(adj, input)
        output = self.linear(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AttentionWeight(nn.Module):
    def __init__(self, nfeature):
        super(AttentionWeight, self).__init__()
        self.l1 = nn.Linear(nfeature, 1)
        
    def forward(self, input, mask=None):
        # 50*16*128 --> 50*16*1
        out = self.l1(input)
        if mask is None:
            out = out
        else:
            out = out.masked_fill(mask == 0, -1e9)
        # 50*16*1 --> 50*1*16
        attn_weights = F.softmax(out, dim=-2).transpose(-2, -1)
        #50*1*16 * 50*16*128 --> 50*1*128
        output = attn_weights@input

        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = self.gc1(x, adj)
        output = F.relu(out)
        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        logpt = F.logsigmoid(input)
        pt = Variable(logpt.data.exp())

        loss = -1 * target*(1-pt)**self.gamma * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()


class LabelAttention(nn.Module):
    def __init__(self, hidden_size, label_embed_size, dropout_rate):
        super(LabelAttention, self).__init__()
        self.l1 = nn.Linear(hidden_size, label_embed_size, bias=False)
        self.tnh = nn.Tanh()
        #self.dropout = nn.Dropout(dropout_rate)
        torch.nn.init.normal(self.l1.weight, 0, 0.03)

    def forward(self, hidden, label_embeds, attn_mask=None):
        # output_1: B x S x H -> B x S x E
        output_1 = self.tnh(self.l1(hidden))
        #output_1 = self.dropout(output_1)

        # output_2: (B x S x E) x (E x L) -> B x S x L
        output_2 = torch.matmul(output_1, label_embeds.t())

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x L -> B x L x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x L x S) @ (B x S x H) -> B x L x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


class Ours(nn.Module):
    def __init__(self, embed_weights, embed_size, freeze_embed, max_len, num_layers, num_heads, forward_expansion,
                 output_size, dropout_rate, label_desc, device, adj_matrix, batchsize, hidden_size, rnn_layer, label_index_1, label_index_2,
                make_sentence, use_gcn=True, lstm=True, pad_idx=0):
        super(Ours, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.adj_matrix = adj_matrix.to(self.device)
        self.register_buffer('label_desc', label_desc)
        self.register_buffer('label_desc_mask', (self.label_desc != self.pad_idx) * 1.0)

        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)

        self.dropout = nn.Dropout(dropout_rate)

        if use_gcn:
            self.label_gcn_1 = GCN(self.embed_size, self.embed_size, self.dropout_rate)
            self.label_gcn_2 = GCN(self.embed_size, self.embed_size, self.dropout_rate)
            self.label_gcn = GCN(self.embed_size, self.embed_size, self.dropout_rate)
            self.label_hierachy = AttentionWeight(embed_size)

        self.label_attn = LabelAttention(hidden_size, embed_size, dropout_rate)
        self.l = nn.Linear(hidden_size*2, hidden_size, bias=False)

        self.fcs = nn.Linear(hidden_size, output_size)
        torch.nn.init.normal(self.fcs.weight, 0, 0.03)
        
        self.layer_norm_0 = nn.LayerNorm(self.embed_size)
        self.layer_norm_1 = nn.LayerNorm(self.embed_size)
        self.layer_norm_2 = nn.LayerNorm(self.embed_size)

        self.label_index_1 = label_index_1.type(torch.bool).to(self.device)
        self.label_index_2 = label_index_2.type(torch.bool).to(self.device)

        self.use_gcn = use_gcn
        self.lstm = lstm
        self.batch_size = batchsize

        if self.lstm:
            self.rnn_layers = rnn_layer
            self.hidden_size = hidden_size
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.rnn_layers,
                               bidirectional=True, dropout=dropout_rate if self.rnn_layers > 1 else 0)
        else:
            self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
            encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                     dim_feedforward=forward_expansion * embed_size,
                                                     dropout=dropout_rate)
            self.encoder = TransformerEncoder(encoder_layers, num_layers)

        self.ms = make_sentence
        if make_sentence:
            self.label_lstm = nn.LSTM(self.embed_size, self.embed_size, num_layers=1, bidirectional=True)
            self.label_fc = nn.Linear(3*self.embed_size, self.embed_size, bias=False)
            self.label_weight = AttentionWeight(3*embed_size)
            torch.nn.init.normal(self.label_fc.weight, 0, 0.03)

    def embed_label_desc(self):
        label_embeds = self.embedder(self.label_desc).transpose(1, 2).matmul(self.label_desc_mask.unsqueeze(2))
        label_embeds_0 = torch.div(label_embeds.squeeze(2), torch.sum(self.label_desc_mask, dim=-1).unsqueeze(1))
        
        if self.ms:
            label_embeds = self.embedder(self.label_desc)
            nb_code = label_embeds.shape[0]
            hidden = self.init_hidden(2, nb_code, self.embed_size, 1)
            label_embeds_input = label_embeds.permute(1, 0, 2)
            label_embeds_lstm, _ = self.label_lstm(label_embeds_input, hidden)
            label_embeds_lstm = label_embeds_lstm.permute(1, 0, 2)
            label_embeds = torch.cat([label_embeds, label_embeds_lstm], axis=-1)
            label_embeds = self.label_weight(label_embeds, self.label_desc_mask.unsqueeze(2)).squeeze(1)
            label_embeds_0 = self.label_fc(label_embeds)

        nb1 = self.label_index_1.shape[0]
        nb2 = self.label_index_2.shape[0]
        nb0 = self.label_index_2.shape[1]

        label_embeds_1 = torch.zeros((nb0, self.embed_size)).to(self.device)
        label_embeds_2 = torch.zeros((nb0, self.embed_size)).to(self.device)

        for i in range(nb1):
            lines = label_embeds_0[self.label_index_1[i]].view(-1, self.embed_size)
            label_embeds_1[self.label_index_1[i, :, 0], :] = torch.mean(lines, axis=0)

        for i in range(nb2):
            lines = label_embeds_0[self.label_index_2[i]].view(-1, self.embed_size)
            label_embeds_2[self.label_index_2[i, :, 0], :] = torch.mean(lines, axis=0)

        if self.use_gcn:
            label_embeds_0_gcn = self.label_gcn(label_embeds_0, self.adj_matrix)
            label_embeds_1_gcn = self.label_gcn_1(label_embeds_1, self.adj_matrix)
            label_embeds_2_gcn = self.label_gcn_2(label_embeds_2, self.adj_matrix)
            
            label_embeds_0 = self.layer_norm_0(label_embeds_0 + label_embeds_0_gcn)
            label_embeds_1 = self.layer_norm_1(label_embeds_1 + label_embeds_1_gcn)
            label_embeds_2 = self.layer_norm_2(label_embeds_2 + label_embeds_2_gcn)
            
            label_embeds = torch.cat(
                [label_embeds_0.unsqueeze(1), label_embeds_1.unsqueeze(1), label_embeds_2.unsqueeze(1)],
                dim=1)
            label_embeds = self.label_hierachy(label_embeds).squeeze(1)

        return label_embeds

    def init_hidden(self, n_directions, batch_size, hidden_size, rnn_layers):

        h = Variable(torch.zeros(rnn_layers * n_directions, batch_size, hidden_size)).to(self.device)
        c = Variable(torch.zeros(rnn_layers * n_directions, batch_size, hidden_size)).to(self.device)
        return h, c

    def forward(self, inputs, lengths, targets=None):
        # attn_mask: B x S -> B x S x 1
        attn_mask = (inputs != self.pad_idx).unsqueeze(2).to(self.device)
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S

        if self.lstm:
            hidden = self.init_hidden(2, self.batch_size, self.hidden_size, self.rnn_layers)
            embeds = self.embedder(inputs)

            embeds = self.dropout(embeds)
            embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            encoded_inputs, _ = self.rnn(embeds, hidden)
            encoded_inputs = pad_packed_sequence(encoded_inputs, total_length=self.max_len)[0]
            encoded_inputs = encoded_inputs.permute(1, 0, 2)

        else:
            embeds = self.pos_encoder(self.embedder(inputs) * math.sqrt(self.embed_size))  # N x S x E
            embeds = self.dropout(embeds)
            embeds = embeds.permute(1, 0, 2)  # S x N x E
            encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # T x N x E
            encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x T x E

        encoded_inputs = self.l(encoded_inputs)
        label_embeds = self.embed_label_desc()
        weighted_outputs, attn_weights = self.label_attn(encoded_inputs, label_embeds, attn_mask=None)

        outputs = self.fcs.weight.mul(weighted_outputs).sum(dim=2).add(self.fcs.bias)

        return outputs, None, attn_weights
import torch
import numpy as np
import copy
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from layers.attention import MultiHeadAttention


class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super(GCNClassifier, self).__init__()
        self.args = args
        self.in_dim = args.hidden_dim
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(self.in_dim, args.num_class)  # 最后得到3分类

    def forward(self, inputs):
        outputs, h_syn, h_sem = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, outputs, h_syn, h_sem


class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super(GCNAbsaModel, self).__init__()
        self.args = args
        self.num_layers = args.num_layers

        # Bert
        if args.emb_type == "bert":
            config = BertConfig.from_pretrained(args.bert_model_dir)
            config.output_hidden_states = True
            self.bert = BertModel.from_pretrained(args.bert_model_dir, config=config, from_tf=False)

        emb_matrix = torch.Tensor(emb_matrix)
        self.emb_matrix = emb_matrix

        self.in_drop = nn.Dropout(args.input_dropout)

        # create embedding layers Glove词嵌入emb
        self.emb = nn.Embedding(args.token_vocab_size, args.emb_dim, padding_idx=0)  # 如果没有emb_matrix就随机生成
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.to(self.args.device), requires_grad=False)

        self.pos_emb = nn.Embedding(args.pos_vocab_size, args.pos_dim, padding_idx=0) \
                                    if args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(args.post_vocab_size, args.post_dim, padding_idx=0) \
                                    if args.post_dim > 0 else None  # position emb

        # rnn layer=Bi-LSTM
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        if self.args.emb_type == 'bert':
            self.dense = nn.Linear(self.in_dim, args.rnn_hidden * 2)
        else:
            self.rnn = nn.LSTM(self.in_dim, args.rnn_hidden, 1, batch_first=True, bidirectional=True)

        # multi-head attention, 句法依赖树提取特征会有错误，因此提出K头余弦相似度，捕捉语义关系来补充语法信息
        self.cos_attn_adj = KHeadAttnCosSimilarity(args.head_num,
                                                    2 * args.hidden_dim,
                                                    args.threshold)

        # gcn layer
        self.gcn_syn = nn.ModuleList()
        self.gcn_sem = nn.ModuleList()
        # gate weight
        self.w_syn = nn.ParameterList()
        self.w_sem = nn.ParameterList()

        # mask gcn
        self.mask_syn_gcn = GCN(args, args.rnn_hidden * 2, args.hidden_dim)
        # PointwiseFeedForward
        self.syn_mean_pool = PointwiseFeedForward(self.args.hidden_dim*2, self.args.hidden_dim)
        self.sem_mean_pool = PointwiseFeedForward(self.args.hidden_dim*2, self.args.hidden_dim)

        # GCN层, syn GCN + sem GCN
        self.gcn_syn.append(GCN(args, args.rnn_hidden * 2, args.hidden_dim))
        self.gcn_sem.append(GCN(args, args.rnn_hidden * 2, args.hidden_dim))
        for i in range(1, self.args.num_layers):
            self.gcn_syn.append(GCN(args, args.hidden_dim, args.hidden_dim))
            self.gcn_sem.append(GCN(args, args.hidden_dim, args.hidden_dim))
            self.w_syn.append(nn.Parameter(
                torch.FloatTensor(args.hidden_dim, args.hidden_dim).normal_(0, 1)))  # 正态分布
            self.w_sem.append(nn.Parameter(
                torch.FloatTensor(args.hidden_dim, args.hidden_dim).normal_(0, 1)))
        # 语义 MHSA + PCT
        self.mhsa_global = MultiHeadAttention(embed_dim=args.rnn_hidden * 2, n_head=args.head_num)
        self.ffn_global = PointwiseFeedForward(args.rnn_hidden * 2, args.hidden_dim, dropout=args.input_dropout)
        # Hierarchical aspect—based attention
        self.syn_attn = TimeWiseAspectBasedAttn(args.hidden_dim, args.num_layers)
        self.sem_attn = TimeWiseAspectBasedAttn(args.hidden_dim, args.num_layers)

        # AM attention，映射到1维
        self.attn = Attention(2 * args.hidden_dim, args.hidden_dim)

        # learnable hyperparameter，最后的 α
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # fully connect Layer
        self.linear = nn.Linear(2 * args.hidden_dim, args.hidden_dim)

    # 拼接三种词嵌入
    def create_embs(self, tok, pos, post):
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)  # input_dropout
        return embs

    def create_bert_embs(self, tok, pos, post, word_idx, segment_ids):
        bert_outputs = self.bert(tok, token_type_ids=segment_ids)
        feature_output = bert_outputs[0]
        word_embs = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_idx)])
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=-1)
        embs = self.in_drop(embs)
        return embs

    # Bi-LSTM，input：embs, length, embs.size(0)
    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(self.args, batch_size, 1, True)  # 1=一层RNN
        rnn_inputs = pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True)  # 输入压缩为可变长序列。先打包再填充
        rnn_outputs, (_, _) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def create_adj_mask(self, rnn_hidden):
        score_mask = torch.matmul(rnn_hidden, rnn_hidden.transpose(-2, -1))  # 句子自己和自己相乘，也就是句子里每个词都和别的词乘一遍，做相似度计算呢？
        # from torchvision import transforms
        # unloader = transforms.ToPILImage()
        # for i, s in enumerate(score_mask):
        #     image = s.cpu().clone()  # clone the tensor
        #     # image = image.squeeze(0)  # remove the fake batch dimension
        #     image = unloader(image)
        #     image.save('score_mask_{}.jpg'.format(i))
        # for hid, s in zip(rnn_hidden, score_mask):
            # h_len = (hid != 0).size(0)
            # s_len = (s != 0).size(0)
            # h_len = torch.sum(hid != 0, dim=0)[0]
            # s_len = torch.sum(s != 0, dim=0)[0]
            # assert h_len == s_len
        score_mask = (score_mask == 0)  # =0的值为True
        return score_mask

    def graph_comm(self, h0, w, h1, score_mask):
        # H = softmax(h0 * w * h1) 公式(13、14)
        H = torch.matmul(h0, w)
        H = torch.matmul(H, h1.transpose(-2, -1))
        H = H.masked_fill(score_mask, -1e10)  # masked
        b = ~score_mask[:, :, 0:1]
        H = F.softmax(H, dim=-1) * b.float()

        # h = h0 + H * h1 公式(11、12)(15、16)
        h = h0 + torch.matmul(H, h1)
        return h

    def Dense(self, inputs, seq_lens):
        # padd
        inputs = self.dense(inputs)
        inputs_unpad = pack_padded_sequence(inputs, seq_lens.cpu(), batch_first=True)
        outputs, _ = pad_packed_sequence(inputs_unpad, batch_first=True)
        return outputs

    # mask dep_dist
    def feature_dynamic_mask(self, text, asp, asp_mask=None, distances_input=None):
        texts = text.cpu()  # batch_size x seq_len
        asps = asp.cpu()  # batch_size x aspect_len
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
            mask_len = self.args.syn_srd
        if asp_mask is not None:
            asp_mask = asp_mask.cpu()
            mask_len = self.args.sem_srd
        masked_text_vec = np.ones((text.size(0), text.size(1), self.args.rnn_hidden),
                                          dtype=np.float32)  # batch_size x seq_len x rnn hidden size*2
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):  # For each sample
            if distances_input is None:
                # asp_len = np.count_nonzero(asps[asp_i])  # Calculate aspect length
                asp_len = torch.count_nonzero(asps[asp_i])
                try:
                    # asp_begin = np.argwhere(texts[text_i] == asps[asp_i][0])[0][0]
                    asp_begin = torch.nonzero(asp_mask[asp_i])[0]
                except:
                    continue
                # Mask begin -> Relative position of an aspect vs the mask
                if asp_begin >= mask_len:
                    mask_begin = asp_begin - mask_len
                else:
                    mask_begin = 0
                for i in range(mask_begin):  # Masking to the left
                    masked_text_vec[text_i][i] = np.zeros(self.args.rnn_hidden, dtype=np.float)
                for j in range(asp_begin + asp_len + mask_len, text.size(1)):  # Masking to the right
                    masked_text_vec[text_i][j] = np.zeros(self.args.rnn_hidden, dtype=np.float)
            else:
                distances_i = distances_input[text_i][:len(texts[1])]  # 按行取
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        masked_text_vec[text_i][i] = np.zeros(self.args.rnn_hidden, dtype=np.float)

        masked_text_vec = torch.from_numpy(masked_text_vec)
        return masked_text_vec.to(self.args.device)

    # ############## Model ################
    def forward(self, inputs):
        if self.args.emb_type == "bert":
            tok, asp, pos, head, dep, post, asp_mask, length, adj, word_idx, segment_ids, dist = inputs
            embs = self.create_bert_embs(tok, pos, post, word_idx, segment_ids)
            hidden = self.Dense(embs, length)
        else:
            tok, asp, pos, head, dep, post, asp_mask, length, adj, dist = inputs       # unpack inputs
            # 三重embedding
            embs = self.create_embs(tok, pos, post)  # 三种词嵌入：Glove+POS(词性标注)+位置嵌入 (bs,seq_len,emb_dim+pos_dim+post_dim)
            # Bi-LSTM encoding
            hidden = self.encode_with_rnn(embs, length, embs.size(0))  # [batch_size, seq_len, rnn_hidden*2]
            # bi-lism 句子级特征提取
        score_mask = self.create_adj_mask(hidden)  # score_mask=0的值为True，等于0表示词之间完全不相关嘛，句子之间做相似度计算呢？

        # cosine adj matrix、KHeadAttnCosSimilarity 捕获语义相似度
        cos_adj = self.cos_attn_adj(hidden, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        cos_adj = torch.sum(cos_adj, dim=1) / self.args.head_num  # 4头求平均，公式（6）

        h_syn = []  # 保存三层GCN的输出(保存MHSA
        h_sem = []
        h_syn_mask = []  # 保存mask的输出

        # GCN encoding
        h_syn.append(self.gcn_syn[0](adj, hidden, score_mask, first_layer=True))  # 句法图和BiLSTM输出，一起输入三层SynGCN
        # semGCN origin
        # h_sem.append(self.gcn_sem[0](cos_adj, hidden, score_mask, first_layer=True))  # 语法图和BiLSTM输出，一起输入三层SemGCN
        # 改成MHSA+PCT ############
        h_sem_demo, _ = self.mhsa_global(hidden, hidden)
        h_sem.append(self.ffn_global(h_sem_demo))
        # mask sem local + global
        if self.args.local_text_focus == 'sem_cdm':
            masked_sem_vec = self.feature_dynamic_mask(hidden, asp, asp_mask)
            local_h_sem = torch.mul(h_sem[0], masked_sem_vec)
        h_sem_global_local = torch.cat((h_sem[0], local_h_sem), dim=-1)
        h_sem[0] = self.syn_mean_pool(h_sem_global_local)
        # mask syn local + global
        if self.args.local_dist_focus == 'syn_cdm':
            masked_syn_vec = self.feature_dynamic_mask(hidden, asp, dist)
            local_h_syn = torch.mul(h_syn[0], masked_syn_vec)
        h_syn_global_local = torch.cat((h_syn[0], local_h_syn), dim=-1)  # 方案2
        h_syn[0] = self.syn_mean_pool(h_syn_global_local)
        # origin ############################ 还有两层
        for i in range(self.args.num_layers - 1):
            # graph communication layer
            h_syn_ = self.graph_comm(h_syn[i], self.w_syn[i], h_sem[i], score_mask)
            h_sem_ = self.graph_comm(h_sem[i], self.w_sem[i], h_syn[i], score_mask)

            h_syn.append(self.gcn_syn[i + 1](adj, h_syn_, score_mask, first_layer=False))
            h_sem.append(self.gcn_sem[i + 1](cos_adj, h_sem_, score_mask, first_layer=False))

        h_syn = torch.stack(h_syn, dim=0)
        h_sem = torch.stack(h_sem, dim=0)

        # time-wise aspect-based attention 基于时间方面的注意
        h_syn_final = self.syn_attn(h_syn[:-1], h_syn[-1], asp_mask, score_mask)
        h_sem_final = self.sem_attn(h_sem[:-1], h_sem[-1], asp_mask, score_mask)

        # h = torch.cat((h_syn_final, h_sem_final), dim=-1)
        h = self.alpha * h_syn_final + (1 - self.alpha) * h_sem_final

        # linear
        outputs = torch.tanh(self.linear(h))  # F.tanh
        # activa = nn.LeakyReLU(0.1)
        # outputs = activa(self.linear(h))
        return outputs, h_syn_final, h_sem_final


# 构造h0，c0的0矩阵
def rnn_zero_state(args, batch_size, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, args.rnn_hidden)
    h0 = c0 = torch.zeros(*state_shape)
    return h0.to(args.device), c0.to(args.device)


class GCN(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GCN, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.Linear(input_dim, output_dim, bias=False)  # 是否加bias

    def forward(self, adj, inputs, score_mask, first_layer=True):
        # gcnlog  ASGCN公式（2）（3）
        # 画出邻接矩阵
        # from torchvision import transforms
        # unloader = transforms.ToPILImage()
        # for i, s in enumerate(adj):
        #     image = s.cpu().clone()  # clone the tensor
        #     # image = image.squeeze(0)  # remove the fake batch dimension
        #     image = unloader(image)
        #     image.save('./pictures/adj_{}.jpg'.format(i))
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj 度矩阵：表示连接节点数
        Ax = adj.bmm(inputs)
        AxW = self.W(Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        out = gAxW if first_layer else self.drop(gAxW)
        return out  # （16,41,204)


# 提取语义关系，补充句法信息
class KHeadAttnCosSimilarity(nn.Module):
    def __init__(self, head_num, input_dim, threshold):
        super(KHeadAttnCosSimilarity, self).__init__()
        assert (input_dim / head_num) != 0
        self.d_k = int(input_dim // head_num)
        self.head_num = head_num
        self.threshold = threshold

        self.mapping = nn.Linear(input_dim, input_dim)
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(input_dim, input_dim))
                                      for _ in range(2)])

    # create cosine similarity adj matrix
    # sem_embs: [batch_size, head_num, seq_len, d_k]
    # score_mask: [batch_size, head_num, seq_len, seq_len]
    # 公式（7）
    def create_cos_adj(self, sem_embs, score_mask):
        seq_len = sem_embs.size(2)

        # calculate the cosine similarity between each words in the sentences
        a = sem_embs.unsqueeze(3)  # [batch_size, head_num, seq_len, 1, d_k]
        b = sem_embs.unsqueeze(2).repeat(1, 1, seq_len, 1, 1)  # [batch_size, head_num, seq_len, seq_len, d_k]
        cos_similarity = F.cosine_similarity(a, b, dim=-1)  # [batch_size, head_num, seq_len, seq_len]
        cos_similarity = cos_similarity * (~score_mask).float()  # mask,score_mask=True的地方mask为0,反转false=0
        # 消除不是句子成分的影响，pad的影响
        # keep the value larger than threshold as the connection
        cos_adj = (cos_similarity > self.threshold).float()
        # from torchvision import transforms
        # unloader = transforms.ToPILImage()
        # for i, s in enumerate(cos_adj):
        #     for j, c in enumerate(s):
        #         image = c.cpu().clone()  # clone the tensor
        #         # image = image.squeeze(0)  # remove the fake batch dimension
        #         image = unloader(image)
        #         image.save('cos_adj_{}_{}.jpg'.format(i, j))
        return cos_adj  # 公式（7）的结果ai,j

    # attn = ((QW)(KW)^T)/sqrt(d)
    # query, key: [batch_size, seq_len, hidden_dim]
    # score_mask: [batch_size, head_num, seq_len, seq_len]
    def attention(self, query, key, score_mask):
        nbatches = query.size(0)
        seq_len = query.size(1)

        query, key = [l(x).view(nbatches, seq_len, self.head_num, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, head_num, seq_len, seq_len]

        scores = scores.masked_fill(score_mask, -1e10)  # score=True的地方（即为0的地方）用-1e10代替,softmax一下≈0
        p_attn = F.softmax(scores, dim=-1)  # [batch_size, head_num, seq_len, seq_len]  出来的权重都比较评价呢，没有起到atten的作用

        b = ~score_mask[:, :, :, 0:1]  # Fasle为0，score_mask正方体，提取第一个：后面的长度都一样
        p_attn = p_attn * b.float()  # [batch_size, head_num, seq_len, 1] 广播机制
        return p_attn  # [batch_size, head_num, seq_len, seq_len]

    # embs: [batch_size, seq_len, input_dim]
    # score_mask: [batch_size, seq_len, seq_len]
    def forward(self, embs, score_mask):
        batch_size = embs.size(0)
        seq_len = embs.size(1)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)  # [batch_size, head_num, seq_len, seq_len] 复制head_num个score_mask

        embs_mapped = self.mapping(embs)  # 公式（4），过Linear，[batch_size, seq_len, input_dim]
        sem_embs = embs_mapped.view(batch_size, seq_len, self.head_num, self.d_k)\
                            .transpose(1, 2)  # [batch_size, head_num, seq_len, d_k] 分成四个头

        K_head_cosine = self.create_cos_adj(sem_embs, score_mask)   # 公式（7）[batch_size, head_num, seq_len, seq_len]

        # multi-head attn for embs_mapped
        attn = self.attention(embs_mapped, embs_mapped, score_mask)  # 公式（8）

        K_head_attn_cosine = K_head_cosine * attn
        return K_head_attn_cosine  # 返回公式（6），未求平均前


# Hierarchical aspect—based attention
class TimeWiseAspectBasedAttn(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(TimeWiseAspectBasedAttn, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim).normal_(0, 1))
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    # context [batch_size, seq_len, hidden_dim]
    # aspect [batch_size, 1, hidden_dim]
    def forward(self, h, h_T, asp_mask, score_mask):
        h_T_ = h_T.unsqueeze(0).repeat(self.num_layers - 1, 1, 1, 1)  # [num_layers - 1, batch_size, seq_len, hidden_dim]
        a0 = torch.sum(h * h_T_, dim=-1, keepdim=True)
        a0 = F.softmax(a0, dim=0)  # 第0维，1层和3层权重，2层和3层权重，哪个大

        h_weighted = torch.sum(a0 * h, dim=0) + h_T  # 公式（20）

        # avg pooling asp and context fearure
        # mask: [batch_size, seq_len]
        asp_wn = asp_mask.sum(dim=1, keepdim=True)  # asp_len  [batch_size, 1]
        asp_mask = asp_mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim)  # mask for h:[batch_size, seq_len, hidden_dim]

        aspect = (h_weighted * asp_mask).sum(dim=1) / asp_wn  # [batch_size, hidden_dim]  公式（21）
        context = h_weighted * (asp_mask == 0).float()  # [batch_size, seq_len, hidden_dim] 公式（22）

        # aspect based attn
        # aspect x self.W x context
        a1 = torch.matmul(aspect.unsqueeze(1), self.W)  # [batch_size, 1, hidden_dim] 公式（23）
        a1 = torch.matmul(a1, context.transpose(1, 2)).transpose(1, 2)  # [batch_size, seq_len, 1]
        a1 = torch.softmax(a1.masked_fill(score_mask[:, :, 0:1], -1e10), dim=1)

        # weighted and add
        context_weighted_vec = torch.sum(a1 * context, dim=1)  # [batch_size, hidden_dim]  公式（24），求和没有除以长度

        output = torch.cat((context_weighted_vec, aspect), dim=-1)  # [batch_size, 2 * hidden_dim]
        return output


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, mask):
        w = self.project(z)
        w = w.masked_fill(mask[:, :, 0:1], -1e10)
        w = torch.softmax(w, dim=1)
        return w * z, w


# class MultiHeadAttention(nn.Module):
#     # d_model:hidden_dim，h:head_num
#     def __init__(self, args, head_num, hidden_dim, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         assert hidden_dim % head_num == 0
#
#         self.d_k = int(hidden_dim // head_num)
#         self.head_num = head_num
#         self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_dim, hidden_dim)) for _ in range(2)])
#         self.dropout = nn.Dropout(p=dropout)
#
#     def attention(self, query, key, score_mask, dropout=None):
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         if score_mask is not None:
#             scores = scores.masked_fill(score_mask.unsqueeze(1), -1e9)  # 在mask为True时，用-1e9填充张量元素。
#         b = ~(score_mask.unsqueeze(1)[:, :, :, 0:1])
#         p_attn = F.softmax(scores, dim=-1) * b.float()
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         return p_attn
#
#     def forward(self, query, key, score_mask):
#         nbatches = query.size(0)
#         query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linears, (query, key))]
#         attn = self.attention(query, key, score_mask, dropout=self.dropout)
#         output = torch.bmm(attn, query)
#         output = torch.cat(torch.split(output, nbatches, dim=0), dim=-1)
#         output = self.linears(output)
#         output = self.dropout(output)
#         return output, attn


class PointwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None, d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output
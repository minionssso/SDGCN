import json
import random
import spacy
import torch
import numpy as np
from spacy.tokens import Doc
from tree import Tree, head_to_tree, tree_to_adj
from torch.nn.utils.rnn import pad_sequence
import networkx as nx


class DataLoader(object):
    def __init__(self, filename, batch_size, args, dicts):
        self.batch_size = batch_size
        self.args = args
        self.dicts = dicts

        with open(filename) as infile:
            data = json.load(infile)
        
        # preprocess data
        data = self.preprocess(data, dicts)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]  # split batchs
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, dicts):  # data:每个list带有5个dict{token,pos,head,deprel,asp} TODO 可以再加一个dep_dist
        
        processed = []
        for d in data:
            for aspect in d['aspects']:
                # word token
                tok = list(d['token'])
                if self.args.lower:
                    tok = [t.lower() for t in tok]
                asp = list(aspect['term'])  # aspect
                terms_id = [aspect['from']] if aspect['from'] == (aspect['to']-1) else [aspect['from'], aspect['to']-1]
                label = aspect['polarity']  # label
                pos = list(d['pos'])        # pos
                head = list(d['head'])      # head
                dep = list(d['deprel'])     # deprel
                length = len(tok)           # real length
                # dep_dist
                _, dist = calculate_dep_dist(tok, asp, terms_id)
                # position
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i-aspect['to']+1 for i in range(aspect['to'], length)]
                # mask of aspect, asp的位置=1
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]

                # map to ids
                if self.args.emb_type == "glove":
                    tok = map_to_ids(tok, dicts['token'])
                    asp = map_to_ids(asp, dicts['token'])
                elif self.args.emb_type == "bert":
                    tok, asp, segment_ids, word_idx, aspect_idx = self.convert_features_bert(tok, asp)

                label = dicts['polarity'][label]
                pos = map_to_ids(pos, dicts['pos'])
                dep = map_to_ids(dep, dicts['dep'])
                head = [int(x) for x in head]
                assert any([x == 0 for x in head])
                post = map_to_ids(post, dicts['post'])

                if self.args.emb_type == "glove":
                    assert len(tok) == length \
                            and len(pos) == length \
                            and len(head) == length \
                            and len(post) == length \
                            and len(mask) == length \
                            and len(dist) == length
                elif self.args.emb_type == "bert":
                    assert len(pos) == length \
                            and len(head) == length \
                            and len(post) == length \
                            and len(mask) == length \
                            and len(dist) == length

                if self.args.emb_type == "glove":
                    processed += [(tok, asp, pos, head, dep, post, mask, length, dist, label)]
                elif self.args.emb_type == "bert":
                    processed += [(tok, asp, pos, head, dep, post, mask, length, word_idx, segment_ids, dist, label)]

        return processed

    def convert_features_bert(self, sentence, aspect):
        """
        BERT features.
        convert sentence to feature.
        """
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0

        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []

        for word in sentence:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)

        # aspect
        for word in aspect:
            word_aspect_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens = [cls_token] + tokens + [sep_token]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
        word_indexer = [i + 1 for i in word_indexer]
        aspect_indexer = [i + 1 for i in aspect_indexer]

        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens)

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(sentence)
        assert len(aspect_indexer) == len(aspect)

        # 句子后面拼上aspect, segment_idx是bert中用于表示句子idx的标志
        input_cat_ids = input_ids + input_aspect_ids[1:]  # [cls] sen [sep] asp [sep]
        segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

        # tok, asp, seg, word_idx, asp_idx
        return input_cat_ids, input_aspect_ids, segment_ids, word_indexer, aspect_indexer

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        
        # sort all fields by lens for easy RNN operations
        batch, orig_idx = sort_all(batch, batch[7])  # token, seq_len

        # convert to tensors, 填充为同batch一样长
        tok = get_long_tensor(batch[0], batch_size).to(self.args.device)
        asp = get_long_tensor(batch[1], batch_size).to(self.args.device)
        pos = get_long_tensor(batch[2], batch_size).to(self.args.device)
        head = get_long_tensor(batch[3], batch_size).to(self.args.device)
        dep = get_long_tensor(batch[4], batch_size).to(self.args.device)
        post = get_long_tensor(batch[5], batch_size).to(self.args.device)
        mask = get_float_tensor(batch[6], batch_size).to(self.args.device)
        length = torch.LongTensor(batch[7]).to(self.args.device)
        if self.args.emb_type == "bert":
            word_idx = get_long_tensor(batch[8], batch_size).to(self.args.device)
            segment_ids = get_long_tensor(batch[9], batch_size).to(self.args.device)
        dist = get_float_tensor(batch[-2], batch_size).to(self.args.device)
        label = torch.LongTensor(batch[-1]).to(self.args.device)

        def inputs_to_tree_reps(maxlen, head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(l.size(0))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=self.args.loop).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            return adj

        maxlen = max(length)
        adj = torch.tensor(inputs_to_tree_reps(maxlen, head, tok, length)).to(self.args.device)

        if self.args.emb_type == "glove":
            return (tok, asp, pos, head, dep, post, mask, length, adj, dist, label)
        elif self.args.emb_type == "bert":
            return (tok, asp, pos, head, dep, post, mask, length, adj, word_idx, segment_ids, dist, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else 1 for t in tokens]  # the id of [UNK] is ``1''
    return ids


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


class WhitespaceTokenizer(object):
    # 重写spcy的分词（空格完成分词）
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load("en_core_web_sm")  # 得到句法依赖树的工具

nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


# 计算依赖树词距离
def calculate_dep_dist(tok, aspect, terms_id):
    sent = ' '
    sent = sent.join(tok)
    doc = nlp(sent)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    # term_ids = [0] * len(aspect)
    for token in doc:  # 遍历句子中的token
        for child in token.children:  # 遍历所有token，提取和边child的关系
            edges.append(('{}_{}'.format(token.lower_, token.i),
                          '{}_{}'.format(child.lower_, child.i)))

    graph = nx.Graph(edges)

    dist = [0.0]*len(doc)
    text = [0]*len(doc)
    for i, word in enumerate(doc):
        source = '{}_{}'.format(word.lower_, word.i)
        sum = 0
        for term_id, term in zip(terms_id, aspect):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph, source=source, target=target)  # 求最短路径长度
            except:
                sum += len(doc)  # No connection between source and target
        dist[i] = sum/len(aspect)  # 多个asp token分别和句子token之间的距离，再除以asp token的数量
        text[i] = word.text
    return text, dist
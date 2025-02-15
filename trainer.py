import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn import GCNClassifier
import math
import adabound
from copy import deepcopy
from transformers import AdamW
import torch.optim.adam as Adam
from models.semmhsa import SemMHSAClassifier
from models.syngcn import SynGCNClassifier
from models.dssgcn import DSSGCNClassifier


class GCNTrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        if args.model_name == 'syngcn':
            self.model = SynGCNClassifier(args, emb_matrix=emb_matrix).to(self.args.device)
        if args.model_name == 'semmhsa':
            self.model = SemMHSAClassifier(args, emb_matrix=emb_matrix).to(self.args.device)
        if args.model_name == 'dssgcn':
            self.model = DSSGCNClassifier(args, emb_matrix=emb_matrix).to(self.args.device)
        # self.model = GCNClassifier(args, emb_matrix=emb_matrix).to(self.args.device)
        # self.model = self.args.model_class(args, emb_matrix=emb_matrix).to(self.args.device)
        self.metric = 0
        if args.emb_type == 'bert':
            bert_model = self.model.gcn_model.bert  # BertModel.from_pretrained(args.bert_model_dir, config=config, from_tf=False)
            bert_params_dict = list(map(id, bert_model.parameters()))
            base_params = filter(lambda p: id(p) not in bert_params_dict, self.model.parameters())
            self.parameters = [
                {"params": base_params},
                {"params": bert_model.parameters(), "lr": args.bert_lr},
            ]
        else:
            self.parameters = self.model.parameters()
        # if args.emb_type == 'glove':
        #     self._reset_params()

        self.optimizer = AdamW(
            self.parameters, lr=args.lr, weight_decay=args.l2reg)
        # self.optimizer = Adam(
        #     self.parameters, lr=args.lr, weight_decay=args.l2reg, amsgrad=True)
        # self.optimizer = torch.optim.SGD(self.parameters, lr=args.lr, momentum=0.9, weight_decay=args.l2reg)
        # self.optimizer = adabound.AdaBound(self.parameters, lr=args.lr, final_lr=args.final_lr)

        # new_lr = old_lr * gamma

    # load model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.args,
        }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.args.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    # train
    def update(self, batch):
        if self.args.emb_type == "glove":
            inputs = batch[0:10]  # 去除polarity tensor的输入
        elif self.args.emb_type == "bert":
            inputs = batch[0:12]
        label = batch[-1]  # 最后一个就是polarity

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, gcn_outputs, h0, h1 = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        # ##################################
        # class_weights = self.calculate_weights(label)
        # class_weights = torch.FloatTensor(class_weights).to(self.args.device)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        # loss = criterion(logits, label)
        # #####################################
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]

        # backward
        loss.backward()
        self.optimizer.step()
        return loss.data, acc
    # eval
    def predict(self, batch):
        if self.args.emb_type == "glove":
            inputs = batch[0:10]
        elif self.args.emb_type == "bert":
            inputs = batch[0:12]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, gcn_outputs, h0, h1 = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        # #####################################
        # class_weights = self.calculate_weights(label)
        # class_weights = torch.FloatTensor(class_weights).to(self.args.device)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        # loss = criterion(logits, label)
        # #####################################
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()

        return loss.data, acc, predictions, \
               label.data.cpu().numpy().tolist(), predprob, \
               gcn_outputs.data.cpu().numpy()

    def step_decay(self, epoch):
        initial_lr = self.args.lr
        drop = 0.5
        epochs_drop = 5.0
        lrate = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def calculate_weights(self, t):
        t = t.cpu()

        batch_size = t.shape[0]
        pos_num, neg_num, ner_num = 0, 0, 0  # 0:pos; 1:neg; 2:ner
        for i in t:
            pos_num += torch.sum(i == 0)
            neg_num += torch.sum(i == 1)
            ner_num += torch.sum(i == 2)
        pos_alpha = 1 - pos_num / batch_size
        neg_alpha = 1 - neg_num / batch_size
        ner_alpha = 1 - ner_num / batch_size
        class_weights = [pos_alpha, neg_alpha, ner_alpha]
        return class_weights

    def cross_entropy_error(self, y, t):
        y = y.cpu()
        t = t.cpu()

        batch_size = y.shape[0]
        pos_num, neg_num, ner_num = 0, 0, 0  # 0:pos; 1:neg; 2:ner
        loss_pos, loss_neg, loss_ner = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
        for i, j in enumerate(t):
            if j == 0:
                pos_num += torch.sum(j == 0)
                loss_pos += -torch.log(y[i, j] + 1e-7)
            elif j == 1:
                neg_num += torch.sum(j == 1)
                loss_neg += -torch.log(y[i, j] + 1e-7)
            elif j == 2:
                ner_num += torch.sum(j == 2)
                loss_ner += -torch.log(y[i, j] + 1e-7)
        loss_pos = torch.sum(loss_pos) / pos_num
        loss_neg = torch.sum(loss_neg) / neg_num
        loss_ner = torch.sum(loss_ner) / ner_num

        pos_alpha = 1 - pos_num / batch_size
        neg_alpha = 1 - neg_num / batch_size
        ner_alpha = 1 - ner_num / batch_size
        loss_final = pos_alpha * loss_pos + neg_alpha * loss_neg + ner_alpha * loss_ner
        # loss_final = torch.from_numpy(loss_final)
        return loss_final.to(self.args.device)

    def mask_exp(self, batch):
        # batch = [b.cuda() for b in batch]
        if self.args.emb_type == "glove":
            inputs = batch[0:10]
        elif self.args.emb_type == "bert":
            inputs = batch[0:12]
        label = batch[-1]
        length = inputs[0].size(1)  # num of tokens
        self.model.eval()
        logits, h, h_syn, _ = self.model(inputs)  # conventional procedure   size of h:(1,50) logits:'positive': 0, 'negative': 1, 'neutral': 2
        # for j in range(len(h)):  # 2 batch
        hs = h[0].squeeze(0).cpu().detach().numpy()  # convert to numpy h为正常得分（所有词都没有mask）

        h_w = list(range(length))
        r = [0.00 for _ in range(length)]  # score

        for i in range(length):  # 遍历每一词
            inputs_w = deepcopy(inputs)  # 修改tuple值
            inputs_w[0][0][i] = 0
            _, h_w[i], _, _ = self.model(inputs_w)  # h_w[i]是第i个词被mask掉的得分
            # _, h_w[i], adj = self.model(inputs, flag=True, mask_pos=i)  # h_w[i]是第i个词被mask掉的得分
            h_w[i] = h_w[i][0].squeeze(0).cpu().detach().numpy()
            for dim in range(len(hs)):  # 204维
                r[i] += abs(hs[dim] - h_w[i][dim])
        max_r = max(r)
        r = [r[i] / max_r for i in range(length)]

        print(r)
        return r
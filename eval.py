import os
import random
import argparse
import numpy as np
import pickle
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from utils import helper
import torch
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Laptops', help='[Restaurants, Tweets, Laptops, MAMS]')
parser.add_argument('--save_dir', type=str, default='./saved_models/Glove/Laptops/best_model_lap_77.1875.pt', help='Root dir for saving models.')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--dep_dim',type=int,default=30, help='Deprel embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=204, help='GCN mem dim.')
parser.add_argument('--rnn_hidden', type=int, default=204, help='RNN hidden state size.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--l2reg', type=float, default=1e-5, help='l2 .')
parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=80, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--optimizer', type=str, default='Adma', help='Adma; SGD')

parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--head_num', default=4, type=int, help='head_num must be a multiple of 3')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--input_dropout', type=float, default=0.1, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.2, help='GCN layer dropout rate.')
parser.add_argument('--mhsa_dropout', type=float, default=0.6, help='MHSA layer dropout rate.')
# dist_mask
parser.add_argument('--sem_srd', type=int, default=3, help='set sem SRD')
parser.add_argument('--syn_srd', type=int, default=3, help='set syn SRD')
parser.add_argument('--local_sem_focus', type=str, default='sem_cdm', help='sem_cdm or sem_cdw or n')
parser.add_argument('--local_syn_focus', type=str, default='syn_cdm', help='syn_cdm or syn_cdw or n')
# shortcut
parser.add_argument('--shortcut', type=bool, default=True, help='shortcut or not')
parser.add_argument('--mhgcn', default=True, help='mhgcn or gcn')  # MHGCN--GCN

# bert
parser.add_argument('--emb_type', type=str, default='bert', help='[glove, bert]')
parser.add_argument('--bert_lr', type=float, default=2e-5)
parser.add_argument('--bert_model_dir', type=str, default='./bert_model')
parser.add_argument('--DEVICE', type=int, default=0, help='GPU number')
args = parser.parse_args()

# set device
args.device = torch.device("cuda:{}".format(args.DEVICE) if torch.cuda.is_available() else "cpu")
print("run on {}".format(args.device))

# load contants
dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
token_vocab = dict()
with open(vocab_file, 'rb') as infile:
    token_vocab['i2w'] = pickle.load(infile)
    token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}
emb_file = './dataset/'+args.dataset+'/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(token_vocab['i2w'])
assert emb_matrix.shape[1] == args.emb_dim

args.token_vocab_size = len(token_vocab['i2w'])
args.post_vocab_size = len(dicts['post'])
args.pos_vocab_size = len(dicts['pos'])

dicts['token'] = token_vocab['w2i']

if args.emb_type == "bert":
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    args.tokenizer = tokenizer
    args.tokenizer.model_max_length = 90
    args.emb_dim = 768

# load training set and test set
print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
test_batch = DataLoader('./dataset/'+args.dataset+'/case_study.json', args.batch_size, args, dicts)  # test.json

# create the model
trainer = GCNTrainer(args, emb_matrix=emb_matrix)

print("Loading model from {}".format(args.save_dir))
mdict = torch.load(args.save_dir, map_location=args.device)
print(mdict['config'])
model_dict = trainer.model.state_dict()
pretrained_dict = {k: v for k, v in mdict['model'].items() if k in model_dict}
model_dict.update(pretrained_dict)
trainer.model.load_state_dict(model_dict)

for i, batch in enumerate(test_batch):
    r = trainer.mask_exp(batch)  # case study

print("Evaluating...")
predictions, labels = [], []
test_loss, test_acc, test_step = 0., 0., 0
for i, batch in enumerate(test_batch):
    trainer.model.eval()
    loss, acc, pred, label, _, _ = trainer.predict(batch)
    print('prediction:', pred)
    print('label:', label)
    test_loss += loss
    test_acc += acc
    predictions += pred
    labels += label
    test_step += 1
f1_score = metrics.f1_score(labels, predictions, average='macro')

print("test_loss: {}, test_acc: {}, f1_score: {}".format(test_loss/test_step, test_acc/test_step, f1_score))

#  'polarity': {'positive': 0, 'negative': 1, 'neutral': 2}}



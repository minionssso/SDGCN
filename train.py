import os
import sys
import warnings
import six
import torch
import pickle
import fitlog
import random
import argparse
import numpy as np
from utils import helper
from shutil import copyfile
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from transformers import BertTokenizer

warnings.filterwarnings('ignore')  # 忽略userwarning

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tweets', help='[Restaurants, Tweets, Laptops]')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.') # 30
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')  # 30
parser.add_argument('--dep_dim', type=int, default=30, help='Deprel embedding dimension')  # 30
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=300, help='GCN mem dim.')  # 204
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')  # 204
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')
parser.add_argument('--lower', type=bool, default=True, help='Lowercase all words.')
parser.add_argument('--direct', type=bool,  default=False, help='Digraph')
parser.add_argument('--loop', type=bool, default=True, help='Self loop')
parser.add_argument('--l2reg', type=float, default=1e-5, help='l2 .')
parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=40, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optimizer', type=str, default='Adma', help='Adma; SGD')
parser.add_argument('--load_model', type=bool, default=False, help='load param or not')
parser.add_argument('--load_model_path', type=str, default='./saved_models/best_model.pt', help='load model path')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--seed', type=int, default=990, help='random seed')  # random.randint(0, 10000) 990
parser.add_argument('--input_dropout', type=float, default=0.1, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--mhsa_dropout', type=float, default=0.6, help='MHSA layer dropout rate.')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--head_num', type=int, default=8, help='head_num must be divisible by hidden_dim')

# bert
parser.add_argument('--emb_type', type=str, default='bert', help='[glove, bert]')
parser.add_argument('--bert_lr', type=float, default=5e-5, help='5e-5, 3e-5, 2e-5')
# parser.add_argument('--bert_dim', type=int, default=768, help='Bert embedding dimension.')
parser.add_argument('--bert_model_dir', type=str, default='./bert_model', help='Root dir for loading pretrained bert')
parser.add_argument('--DEVICE', type=int, default=0, help='The number of GPU')
# dist_mask
parser.add_argument('--sem_srd', type=int, default=5, help='set sem SRD')
parser.add_argument('--syn_srd', type=int, default=5, help='set syn SRD')
parser.add_argument('--local_sem_focus', type=str, default='sem_cdm', help='sem_cdm or cdw or n')
parser.add_argument('--local_syn_focus', type=str, default='syn_cdm', help='syn_cdm or cdw or n')
# shortcut
parser.add_argument('--shortcut', type=bool, default=True, help='shortcut or not')
args = parser.parse_args()

# set device
args.device = torch.device("cuda:{}".format(args.DEVICE) if torch.cuda.is_available() else "cpu")
print("run on {}".format(args.device))

# if you want to reproduce the result, fix the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
helper.print_arguments(args)


# load contants
dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
token_vocab = dict()
with open(vocab_file, 'rb') as infile:
    token_vocab['i2w'] = pickle.load(infile)
    token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}

# load dict
emb_file = './dataset/'+args.dataset+'/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(token_vocab['i2w'])
assert emb_matrix.shape[1] == args.emb_dim

args.token_vocab_size = len(token_vocab['i2w'])
args.post_vocab_size = len(dicts['post'])
args.pos_vocab_size = len(dicts['pos'])
args.dep_vocab_size = len(dicts['dep'])

if args.emb_type == "bert":
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    args.tokenizer = tokenizer
    args.tokenizer.model_max_length = 90
    args.emb_dim = 768

fitlog.set_log_dir("logs/Dual_MGCN_MHSA/Tweets")         # TODO 设定日志存储的目录 logs/Rests Laptops Tweets
for arg, value in sorted(six.iteritems(vars(args))):
    fitlog.add_hyper({arg: value})  # 记录ArgumentParser的参数

dicts['token'] = token_vocab['w2i']

# load training set and test set
print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
train_batch = [batch for batch in DataLoader(
                './dataset/'+args.dataset+'/train.json', args.batch_size, args, dicts)]
test_batch = [batch for batch in DataLoader(
                './dataset/'+args.dataset+'/test.json', args.batch_size, args, dicts)]
# 当实例对象通过[]运算符取值时，会调用它的__getitem__()
# create the folder for saving the best models and log file
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + args.log,
                                header="#poch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttest_f1")

trainer = GCNTrainer(args, emb_matrix=emb_matrix)

if args.load_model:
    print('loading model from' + args.load_model_path)
    mdict = torch.load(args.load_model_path, map_location=args.device)
    print(mdict['config'])
    model_dict = trainer.model.state_dict()
    pretrained_dict = {k: v for k, v in mdict['model'].items() if k in model_dict} # and k != 'gcn_model.linear.weight' and k != 'gcn_model.linear.bias'}
    model_dict.update(pretrained_dict)
    trainer.model.load_state_dict(model_dict)

# ################################
train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
test_acc_history = [0.]
adjust_lr_signal = 0
for epoch in range(1, args.num_epoch+1):
    print('\nepoch:%d' %epoch)
    train_loss, train_acc, train_step = 0., 0., 0
    for batch in train_batch:
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:

            print("train_loss: {:1.4f}, train_acc: {:1.4f}".format(train_loss/train_step, train_acc/train_step))

    # eval on test
    print("Evaluating on test set...")
    predictions, labels = [], []
    test_loss, test_acc, test_step = 0., 0., 0
    for batch in test_batch:
        loss, acc, pred, label, _, _ = trainer.predict(batch)
        test_loss += loss
        test_acc += acc
        predictions += pred
        labels += label
        test_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    print("trian_loss: {:1.4f}, test_loss: {:1.4f}, train_acc: {:1.4f}, test_acc: {:1.4f}, "
          "f1_score: {:1.4f}".format(
        train_loss/train_step, test_loss/test_step,
        train_acc/train_step, test_acc/test_step,
        f1_score))

    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
        epoch, train_loss/train_step, test_loss/test_step,
        train_acc/train_step, test_acc/test_step,
        f1_score))

    train_acc_history.append(train_acc/train_step)
    train_loss_history.append(train_loss/train_step)
    test_loss_history.append(test_loss/test_step)

    # save best model
    if epoch == 1 or test_acc/test_step > max(test_acc_history):
        trainer.save(model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"
            .format(epoch, train_loss/train_step, test_loss/test_step,
                    train_acc/train_step, test_acc/test_step, f1_score))
        adjust_lr_signal = 0

        # figlog
        fitlog.add_best_metric({"test": {"best_Acc": test_acc / test_step}})
        fitlog.add_best_metric({"test": {"best_f1": f1_score}})

    if adjust_lr_signal > 5:
        print('stop training')
        break

    test_acc_history.append(test_acc/test_step)
    f1_score_history.append(f1_score)
    adjust_lr_signal += 1

fitlog.finish()
print("Training ended with {} epochs.".format(epoch))
bt_test_acc = max(test_acc_history)
bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
print("best test_acc/f1_score: {}/{}".format(bt_test_acc, bt_f1_score))


# system package
import os

import argparse
import time
import math
import traceback

parser = argparse.ArgumentParser('Running the recommendation task on Last FM dataset')
parser.add_argument('--train_dir', type=str, default='StackOverflow-log')
parser.add_argument('--time_basis', action='store_true', help='Mercer\'s time embedding')
parser.add_argument('--time_bochner', action='store_true', help='non-parametric Bochner time embedding')
parser.add_argument('--time_gaussian', action='store_true', help='Bochner with Gaussian distribution')
parser.add_argument('--time_rand', action='store_true', help='Bochner with uniformly sampled frequencies')
parser.add_argument('--time_pos', action='store_true', help='positional encoding')
parser.add_argument('--time_flex', action='store_true', help='Mercer\'s time embedding with more free parameters')
parser.add_argument('--time_inv_cdf', action='store_true', help='Flexible inverse CDF based Bochner embeddings')

parser.add_argument('--inv_cdf_method', type=str, default='mlp_res', choices=['mlp_res', 'maf', 'iaf', 'NVP'], help='choose the CDF approximation method')

parser.add_argument('--CUDA_device', type=int, default=0, help='GPU bus id')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int, help='Maximum length of the sequence')
parser.add_argument('--hidden_units', default=32, type=int, help='Dimension of the embeddings')
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.000, type=float)
parser.add_argument('--expand_factor', type=int, default=10, help='Degree of expansion used for Mercer\'s time embedding')
parser.add_argument('--time_factor', type=float, default=1, help='(#dimension of time encoding) / (#dimension of embeddings)')
parser.add_argument('--data_idx', type=int, default=1)


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.CUDA_device)

# 3rd party
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# gpu friendly
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

# self-defined
from self_attention.model import ClassModel
from self_attention.util import *

run_folder_name = args.train_dir

if not os.path.isdir(run_folder_name):
    os.makedirs(run_folder_name)
with open(os.path.join(run_folder_name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

data_idx = args.data_idx

def build_train(event_seq, time_seq, max_len, expand=True):
    data_train = {}
    final_data_train = {}
    item_set = defaultdict(int)
    u = 0
    for (ii, tt) in zip(event_seq, time_seq):
        curr_line = []
        for (i, t) in zip(ii, tt):
            curr_line.append((i, t))
            item_set[i] += 1
        curr_line = sorted(curr_line, key=lambda x: x[1])
        for end in range(1, len(curr_line) + 1):
            data_train[u] = curr_line[max(0, end - max_len):end]
            u += 1
            
    final_data_train = data_train
    for u in final_data_train:
        # pass
        max_time = max([x[1] for x in final_data_train[u]])
        final_data_train[u] = [(i, max_time - t) for i, t in final_data_train[u]]
    return final_data_train, len(event_seq), len(item_set), item_set


def user_profile(idx, max_len):
    event_seq = []
    with open('./input_data/so/event-{}-train.txt'.format(idx)) as f:
        for line in f:
            x  = [int(t) for t in line.rstrip().split(' ')]
            event_seq.append(x)

    time_seq = []
    with open('./input_data/so/time-{}-train.txt'.format(idx)) as f:
        for line in f:
            x  = [float(t) for t in line.rstrip().split(' ')]
            time_seq.append(x)


    test_event_seq = []
    with open('./input_data/so/event-{}-test.txt'.format(idx)) as f:
        for line in f:
            x  = [int(t) for t in line.rstrip().split(' ')]
            test_event_seq.append(x)

    test_time_seq = []
    with open('./input_data/so/time-{}-test.txt'.format(idx)) as f:
        for line in f:
            x  = [float(t) for t in line.rstrip().split(' ')]
            test_time_seq.append(x)
            
    data_train, usernum, itemnum, item_count = build_train(event_seq, time_seq, max_len)
    data_test, _, _, _ = build_train(test_event_seq, test_time_seq, max_len)
    return data_train, data_test, usernum, itemnum, item_count

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(data_train, usernum, itemnum, batch_size, maxlen):
    def sample():

        user = np.random.randint(0, usernum)
        while len(data_train[user]) <= 1: user = np.random.randint(0, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
    
        seq_t = np.zeros([maxlen], dtype=np.float32)
        label_t = np.zeros([maxlen], dtype=np.float32)
        
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        
        nxt = data_train[user][-1]
        idx = maxlen - 1

        ts = set(data_train[user])
        for (i, t) in reversed(data_train[user][:-1]):
            seq[idx] = i
            seq_t[idx] = t
            
            pos[idx] = nxt[0]
            label_t[idx] = nxt[1]
            
            if nxt[0] != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                
            nxt = (i, t) 
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg, seq_t, label_t)

    #np.random.seed(SEED)
    max_len = maxlen
    user_b = np.zeros(batch_size, dtype=np.int32)
    seq_b = np.zeros((batch_size, max_len), dtype=np.int32)
    pos_b = np.zeros((batch_size, max_len), dtype=np.int32)
    neg_b = np.zeros((batch_size, max_len), dtype=np.int32)
    seq_tb = np.zeros((batch_size, max_len), dtype=np.float32)
    label_tb = np.zeros((batch_size, max_len), dtype=np.float32)

    for i in range(batch_size):
        user, seq, pos, neg, seq_t, label_t = sample()
        user_b[i] = user
        seq_b[i, :] = seq
        pos_b[i, :] = pos
        neg_b[i, :] = neg
        seq_tb[i, :] = seq_t
        label_tb[i, :] = label_t

    return user_b, seq_b, pos_b, neg_b, seq_tb, label_tb


def iter_function(curr_idx, data_train, usernum, itemnum, batch_size, maxlen):
    def sample(curr_idx):

        user = curr_idx
        seq = np.zeros([maxlen], dtype=np.int32)
    
        seq_t = np.zeros([maxlen], dtype=np.float32)
        label_t = np.zeros([maxlen], dtype=np.float32)
        
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = data_train[user][-1]
        idx = maxlen - 1

        ts = set(data_train[user])
        for (i, t) in reversed(data_train[user][:-1]):
            seq[idx] = i
            seq_t[idx] = t
            
            pos[idx] = nxt[0]
            label_t[idx] = nxt[1]
            
            if nxt[0] != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                
            nxt = (i, t) 
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg, seq_t, label_t)

    #np.random.seed(SEED)
    max_len = maxlen
    user_b = np.zeros(batch_size, dtype=np.int32)
    seq_b = np.zeros((batch_size, max_len), dtype=np.int32)
    pos_b = np.zeros((batch_size, max_len), dtype=np.int32)
    neg_b = np.zeros((batch_size, max_len), dtype=np.int32)
    seq_tb = np.zeros((batch_size, max_len), dtype=np.float32)
    label_tb = np.zeros((batch_size, max_len), dtype=np.float32)

    for i in range(batch_size):
        user, seq, pos, neg, seq_t, label_t = sample(curr_idx)
        curr_idx += 1
        if curr_idx not in data_train:
            curr_idx = 0
        user_b[i] = user
        seq_b[i, :] = seq
        pos_b[i, :] = pos
        neg_b[i, :] = neg
        seq_tb[i, :] = seq_t
        label_tb[i, :] = label_t

    return user_b, seq_b, pos_b, neg_b, seq_tb, label_tb, curr_idx


data_train, data_test, usernum, itemnum, item_count = user_profile(data_idx, args.maxlen)

x = [v[-1][0] for v in data_train.values()]
y = [v[-1][0] for v in data_test.values()]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)


model = ClassModel(usernum, itemnum, args, 
                    take_pos=args.time_pos, 
                    take_time=args.time_bochner, 
                    take_base=args.time_basis, 
                    take_inv=args.time_inv_cdf, 
                    take_flex=args.time_flex,
                    take_rand=args.time_rand, 
                    take_gaussian=args.time_gaussian,
                    inv_method=args.inv_cdf_method,
                    concat=True, take_last=True, expand_factor=args.expand_factor, time_factor=args.time_factor)


expand_dim = int(args.hidden_units * args.expand_factor)

sess.run(tf.global_variables_initializer())

f = open(os.path.join(run_folder_name, 'log.txt'), 'w')

import traceback
T = 0.0
t0 = time.time()
try:
    for epoch in range(1, args.num_epochs):
        num_batch = len(data_train) // args.batch_size
        acc_l, loss_l = [], []
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, seq_t, label_t = sample_function(data_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen)
            acc, loss, _ = sess.run([model.acc, model.loss, model.train_op],
                                    {model.input_seq: seq, model.pos: pos, model.input_t: seq_t,
                                     model.is_training: True})
            acc_l.append(acc)
            loss_l.append(loss)
        if epoch % 1 == 0:
            t1 = time.time() - t0
            T += t1
            m_acc, m_loss = np.mean(acc_l), np.mean(loss_l)
            info ='Train: epoch: {0}, acc: {1:.4f}, loss: {2:.4f}'.format(epoch, m_acc, m_loss)
            print(info)
            f.write(info + '\n')
            f.flush()
            t0 = time.time()
            
        if epoch % 1 == 0:
            acc_l, loss_l = [], []
            curr_idx = 0
            test_batch_size = 663
            num_batch = len(data_test) // test_batch_size
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg, seq_t, label_t, curr_idx = iter_function(curr_idx, data_test, usernum, itemnum, batch_size=test_batch_size, maxlen=args.maxlen)
                pred_label, loss, is_target, acc = sess.run([model.pred_label, model.loss, model.istarget, model.acc],
                                        {model.input_seq: seq, model.pos: pos, model.input_t: seq_t,
                                         model.is_training: False})

                acc = (pos[:, -1] == pred_label).mean()

                acc_l.append(acc)
                loss_l.append(loss)
            m_acc, m_loss = np.mean(acc_l), np.mean(loss_l)
            info = 'Test: epoch: {0}, acc: {1:.4f}, loss: {2:.4f}'.format(epoch, m_acc, m_loss)
            print(info)
            f.write(info + '\n')
            f.flush()
except:
    f.close()
    traceback.print_exc()
    exit(1)

f.close()
print("Done")
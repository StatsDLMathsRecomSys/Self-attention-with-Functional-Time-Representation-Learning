import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('input_data/ml-1m/ml-1m.txt')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i, 0))

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return User, [user_train, user_valid, user_test, usernum, itemnum]

def ml_1m_timed_data_partition(left_zero=True):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('input_data/ml-1m/ratings.dat')
    
    u_index = {}
    i_index = {}
    u_count = defaultdict(int)
    i_count = defaultdict(int)
    
    User_raw = defaultdict(list)
    User_temp = {}
    for line in f:
        u, i, r, t = [int(x) for x in line.rstrip().split('::')]
        u_count[u] += 1
        i_count[i] += 1
        User_raw[u].append((i, t))
    for u in User_raw:
        l = User_raw[u].copy()
        l = sorted(l, key=lambda x: x[1])
        
        if left_zero:
            min_t = min(x[1] for x in l)
            l = [(i, t - min_t) for i, t in l]
        else:
            max_t = min(x[1] for x in l)
            l = [(i, max_t - t) for i, t in l]
        User_temp[u] = l
        
    for u in User_temp:
        # filter out cold-start items
        for (i, t) in User_temp[u]:
            if u_count[u] < 5 or i_count[i] < 5:
                continue
            # remove gaps in the coding
            if u not in u_index:
                u_index[u] = len(u_index) + 1
            if i not in i_index:
                i_index[i] = len(i_index) + 1

            u = u_index[u]
            i = i_index[i]
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append((i, t))
        
    assert(len(u_index) == usernum)
    assert(len(i_index) == itemnum)
    
    # remove rare item
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return User, [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess, left_zero=True):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_t = np.zeros([args.maxlen], dtype=np.float32)
        
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        seq_t[idx] = valid[u][0][1]
        
        idx -= 1
        for (i, t) in reversed(train[u]):
            seq[idx] = i
            seq_t[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set([x[0] for x in train[u]])
        rated.add(0)
        
        item_idx = [test[u][0][0]]
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx, [seq_t])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 2000 == 0:
            print('.')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_with_gap(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_t = np.zeros([args.maxlen], dtype=np.float32)
        
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        seq_t[idx] = valid[u][0][1]
        
        idx -= 1
        train_tape = train[u]
        rated = set([x[0] for x in train_tape])
        rated.add(0)
        
        if len(train_tape) > 30:
            seq_len = len(train_tape)
            idx_list = sorted(random.sample(list(range(seq_len)), seq_len // 2))
            train_tape = [train_tape[idx_list[i]] for i in range(len(idx_list))]
        
        for (i, t) in reversed(train_tape):
            seq[idx] = i
            seq_t[idx] = t
            idx -= 1
            if idx == -1: break
        
        
        item_idx = [test[u][0][0]]
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx, [seq_t])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 2000 == 0:
            print('.')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, args, sess, left_zero=True):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_t = np.zeros([args.maxlen], dtype=np.float32)
        
        idx = args.maxlen - 1
        #seq[idx] = valid[u][0][0]
        #seq_t[idx] = valid[u][0][1]
        #idx -= 1
        for (i, t) in reversed(train[u]):
            seq[idx] = i
            seq_t[idx] = t
            idx -= 1
            if idx == -1: break
        #rated = set(train[u])
        rated = set([x[0] for x in train[u]])
        rated.add(0)
        
        item_idx = [valid[u][0][0]]
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx, [seq_t])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 2000 == 0:
            print ('.')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
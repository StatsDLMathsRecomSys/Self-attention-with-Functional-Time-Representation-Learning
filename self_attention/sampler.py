import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
    
        seq_t = np.zeros([maxlen], dtype=np.float32)
        label_t = np.zeros([maxlen], dtype=np.float32)
        
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for (i, t) in reversed(user_train[user][:-1]):
            seq[idx] = i
            seq_t[idx] = t
            
            pos[idx] = nxt[0]
            label_t[idx] = nxt[1]
            
            if nxt[0] != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                
            nxt = (i, t) 
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg, seq_t, label_t)

    np.random.seed(SEED)
    max_len = maxlen
    while True:
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
        
        result_queue.put((user_b, seq_b, pos_b, neg_b, seq_tb, label_tb))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
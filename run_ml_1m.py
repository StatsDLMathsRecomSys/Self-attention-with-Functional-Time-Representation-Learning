# system package
import os

import argparse
import time
import math
import traceback


parser = argparse.ArgumentParser('Running the recommendation task on Movie-Lens 1M dataset')

parser.add_argument('--train_dir', type=str, default='Movie-Lens-log')
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
parser.add_argument('--maxlen', default=200, type=int, help='Maximum length of the sequence')
parser.add_argument('--hidden_units', default=72, type=int, help='Dimension of the embeddings')
parser.add_argument('--num_blocks', default=3, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--expand_factor', type=int, default=5, help='Degree of expansion used for Mercer\'s time embedding')
parser.add_argument('--time_factor', type=float, default=1, help='(#dimension of time encoding) / (#dimension of embeddings)')

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
from self_attention.sampler import WarpSampler
from self_attention.model import Model
from self_attention.util import *


run_folder_name = args.train_dir

if not os.path.isdir(run_folder_name):
    os.makedirs(run_folder_name)
with open(os.path.join(run_folder_name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


with open(os.path.join(run_folder_name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


User, dataset = ml_1m_timed_data_partition(left_zero=True)
#User, dataset = data_partition('')
user_train, user_valid, user_test, usernum, itemnum = dataset

num_batch = len(user_train) // args.batch_size
cc_new = []
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
    cc_new.append(len(user_train[u]))

print('average sequence length: {0:.3f}'.format(cc / len(user_train)))

# 'basis', 'bochner', 'inv_cdf', 'gaussian', 'pos'
model = Model(usernum, itemnum, args, 
                take_pos=args.time_pos, 
                take_time=args.time_bochner, 
                take_base=args.time_basis, 
                take_rand=args.time_rand, 
                take_gaussian=args.time_gaussian, 
                take_flex=args.time_flex,
                take_inv=args.time_inv_cdf,
                inv_method=args.inv_cdf_method,
                expand_factor=args.expand_factor,
                time_factor=args.time_factor,
                concat=True
                )

sess.run(tf.global_variables_initializer())

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=4)


f = open(os.path.join(run_folder_name, 'log.txt'), 'w')

T = 0.0
t0 = time.time()

test_hist = []
val_hist = []
time_hist = []
epoch_hist = []


try:
    for epoch in range(1, args.num_epochs):
        for step in range(num_batch):
        #for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            #u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg, seq_t, label_t = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg, model.input_t: seq_t,
                                     model.is_training: True})

        if epoch % 100 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_test = evaluate(model, dataset, args, sess)
            t_test_gap = evaluate_with_gap(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print('epoch:{}, time: {}, valid (NDCG@10:{}, HR@10: {}), test (NDCG@10:{}, HR@10: {}), test_gap (NDCG@10:{}, HR@10: {})'.format(
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1], t_test_gap[0], t_test_gap[1]))
            
            test_hist.append(t_test)
            val_hist.append(t_valid)
            time_hist.append(T)
            epoch_hist.append(epoch)

            f.write('epoch: {}, '.format(epoch) + str(t_valid) + ' ' + str(t_test) + ' ' + str(t_test_gap) + '\n')
            f.flush()
            t0 = time.time()
except:
    sampler.close()
    f.close()
    traceback.print_exc()
    exit(1)

f.close()
sampler.close()
print("Done")


print('epoch:{}, time: {}, valid (NDCG@10:{}, HR@10: {}), test (NDCG@10:{}, HR@10: {}), test_gap (NDCG@10:{}, HR@10: {})'.format(
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1], t_test_gap[0], t_test_gap[1]))


test_ndcg = [x[0] for x in test_hist]
test_hit = [x[1] for x in test_hist]
time_df = pd.DataFrame({'epoch': epoch_hist, 'ndcg':test_ndcg, 'hit':test_hit, 'wall_time':time_hist})
time_df.to_csv(os.path.join(run_folder_name, 'time_run_speed_every_100.csv'))

f.close()
sampler.close()
print("Done")



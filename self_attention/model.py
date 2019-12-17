import sys
sys.path.append("..")

from self_attention.modules import *

class Model:
    '''Train with negative sampling signals'''
    def __init__(self, usernum, itemnum, args, 
                 take_time=True, 
                 take_pos=True, 
                 take_base=True, 
                 take_rand=False, 
                 take_gru=False,
                 concat=False, 
                 take_gaussian=False, 
                 take_inv=False, 
                 take_flex=False, 
                 inv_method=None,
                 reuse=None, 
                 expand_factor=3, 
                 time_factor=1):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        self.input_t = tf.placeholder(tf.float32, shape=(None, args.maxlen))
        # self.label_t = tf.placeholder(tf.float32, shape=(None, args.maxlen))
        
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        hidden_units = args.hidden_units
        embed_units = args.hidden_units
        
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            #Positional Encoding
            if take_pos:
                print('Taking pos encoding!')
                self.t, pos_emb_table = embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                    vocab_size=args.maxlen,
                    num_units=int(embed_units * time_factor),
                    zero_pad=False,
                    scale=False,
                    l2_reg=args.l2_emb,
                    scope="dec_pos",
                    reuse=reuse,
                    with_t=True
                )

                if not concat:
                    self.seq += self.t
                else:
                    self.seq = tf.concat([self.seq, self.t], axis=2)
                    hidden_units += int(embed_units * time_factor)

            if take_time:
                print('Taking time encoding!')
                self.t_e, self.t_e_freq = time_encoding(self.input_t, int(embed_units * time_factor), return_weight=True)

                if not concat:
                    self.seq += self.t_e
                else:
                    self.seq = tf.concat([self.seq, self.t_e], axis=2)
                    hidden_units += int(embed_units * time_factor)
                    
            if take_base:
                print('Taking time basis encoding')
                self.t_bs, self.t_bs_freq = basis_time_encode(self.input_t, embed_units, 
                int(embed_units * time_factor), expand_factor, return_weight=True)
                assert(concat)
                print(self.t_bs)
                
                self.seq = tf.concat([self.seq, self.t_bs], axis=2)
                hidden_units += int(embed_units * time_factor)
                
                
            if take_rand:
                print('Taking random sampling each run')
                self.t_rand = rand_time_encode(self.input_t, int(embed_units * time_factor))
                
                if not concat:
                    self.seq += self.t_rand
                else:
                    self.seq = tf.concat([self.seq, self.t_rand], axis=2)
                    hidden_units += int(embed_units * time_factor)
                    
            if take_gaussian:
                print('Taking Gaussian sample each run')
                self.t_normal = gaussian_time_encode(self.input_t, int(embed_units * time_factor))
                
                if not concat:
                    self.seq += self.t_normal
                else:
                    self.seq = tf.concat([self.seq, self.t_normal], axis=2)
                    hidden_units += int(embed_units * time_factor)
                    
            if take_inv:
                print('Taking inverse CDF sample each run')
                self.t_inv = inverse_cdf_time_encode(self.input_t, int(embed_units * time_factor), method=inv_method)
                
                if not concat:
                    self.seq += self.t_inv
                else:
                    self.seq = tf.concat([self.seq, self.t_inv], axis=2)
                    hidden_units += int(embed_units * time_factor)
                

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            if take_gru:
                
                self.seq = tf.keras.layers.GRU(units=hidden_units, return_sequences=True)(self.seq)
                self.seq *= mask
                
            else:
                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_%d" % i):
                        # Self-attention
                        self.seq = multihead_attention(queries=normalize(self.seq),
                                                       keys=self.seq,
                                                       num_units=hidden_units,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope="self_attention")

                        # Feed forward
                        units_shape = [hidden_units, hidden_units]

                        if concat and i == args.num_blocks - 1:
                            units_shape = [hidden_units, embed_units]

                        self.seq = feedforward(normalize(self.seq), num_units=units_shape,
                                               dropout_rate=args.dropout_rate, is_training=self.is_training)

                        self.seq *= mask

            self.seq = normalize(self.seq)
            
        
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, embed_units])

        self.test_item = tf.placeholder(tf.int32, shape=(None))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, -1])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, seq_time):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_t: seq_time, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
    
  
class ClassModel:
    '''Model for classification task'''
    def __init__(self, usernum, itemnum, args, 
                take_time=True, 
                take_pos=True, 
                take_flex=False, 
                take_base=True, 
                take_rand=False, 
                take_inv=False, 
                inv_method=None, 
                concat=False, 
                reuse=None, 
                take_gaussian=False, 
                take_last=False,  
                expand_factor=3, 
                time_factor=1):
        self.is_training = tf.placeholder(tf.bool, shape=())
        #self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        self.input_t = tf.placeholder(tf.float32, shape=(None, args.maxlen))
        # self.label_t = tf.placeholder(tf.float32, shape=(None, args.maxlen))
        
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        #self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        hidden_units = args.hidden_units
        embed_units = args.hidden_units
        
        pos = self.pos
        #neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            #Positional Encoding
            if take_pos:
                print('Taking pos encoding!')
                self.t, pos_emb_table = embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                    vocab_size=args.maxlen,
                    num_units=embed_units,
                    zero_pad=False,
                    scale=False,
                    l2_reg=args.l2_emb,
                    scope="dec_pos",
                    reuse=reuse,
                    with_t=True
                )

                if not concat:
                    self.seq += self.t
                else:
                    self.seq = tf.concat([self.seq, self.t], axis=2)
                    hidden_units += embed_units

            if take_time:
                print('Taking time encoding!')
                self.t_e = time_encoding(self.input_t, embed_units)

                if not concat:
                    self.seq += self.t_e
                else:
                    self.seq = tf.concat([self.seq, self.t_e], axis=2)
                    hidden_units += embed_units
                    
            if take_base:
                print('Taking time basis encoding')
                self.t_bs = basis_time_encode(self.input_t, embed_units, int(embed_units * time_factor), expand_factor)
                assert(concat)
                print(self.t_bs)
                
                self.seq = tf.concat([self.seq, self.t_bs], axis=2)
                hidden_units += int(embed_units * time_factor)
                
            if take_flex:
                print('Taking time basis encoding')
                self.t_bs, self.t_bs_freq = basis_time_encode_flex(self.input_t, embed_units, int(embed_units * time_factor), expand_factor, return_weight=True)
                assert(concat)
                print(self.t_bs)
                
                self.seq = tf.concat([self.seq, self.t_bs], axis=2)
                hidden_units += int(embed_units * time_factor)
                
                
            if take_rand:
                print('Taking random sampling each run')
                self.t_rand = rand_time_encode(self.input_t, embed_units)
                
                if not concat:
                    self.seq += self.t_rand
                else:
                    self.seq = tf.concat([self.seq, self.t_rand], axis=2)
                    hidden_units += embed_units
                    
                    
            if take_gaussian:
                print('Taking Gaussian sample each run')
                self.t_normal = gaussian_time_encode(self.input_t, embed_units)
                
                if not concat:
                    self.seq += self.t_normal
                else:
                    self.seq = tf.concat([self.seq, self.t_normal], axis=2)
                    hidden_units += embed_units
                
                
            if take_inv:
                print('Taking inverse CDF sample each run')
                self.t_inv = inverse_cdf_time_encode(self.input_t, embed_units, method=inv_method)
                
                if not concat:
                    self.seq += self.t_inv
                else:
                    self.seq = tf.concat([self.seq, self.t_inv], axis=2)
                    hidden_units += embed_units
                

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    units_shape = [hidden_units, hidden_units]
                    
                    if concat and i == args.num_blocks - 1:
                        units_shape = [hidden_units, embed_units]
                        
                    self.seq = feedforward(normalize(self.seq), num_units=units_shape,
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    
                    self.seq *= mask

            self.seq = normalize(self.seq)
            
        
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])    
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, embed_units])
        pred_logits = tf.keras.layers.Dense(units=itemnum + 1, activation=None, use_bias=True)(seq_emb) # [N * max_len, embed_units]
        
        # ignore padding items (0)
        if not take_last:
            istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
            self.istarget = istarget
        else:
            istarget = tf.tile(tf.one_hot(args.maxlen - 1, depth=args.maxlen), [tf.shape(self.input_seq)[0]])
            print(istarget.shape)
            self.istarget = istarget
        one_hot_label = tf.one_hot(pos, depth=itemnum + 1)
        test_target = tf.tile(tf.one_hot(args.maxlen - 1, depth=args.maxlen), [tf.shape(self.input_seq)[0]])
        
        single_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=pred_logits)
        print(single_loss)
        self.loss = tf.reduce_sum(single_loss * istarget) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        pred_label = tf.cast(tf.argmax(pred_logits, axis=1), tf.int32)
        
        self.pred_label = tf.reshape(pred_label, [-1, args.maxlen])[:, -1]
        last_pos = self.pos[:, -1]
        tf.summary.scalar('loss', self.loss)
        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred_label, last_pos), tf.float32))
        self.test_acc = self.acc

        if reuse is None:
            #tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            pass
            #tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, seq, item_idx, seq_time):
        return sess.run(self.pred_label,
                        {self.input_t: seq_time, self.input_seq: seq, self.is_training: False})
# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs


def multihead_attention(queries,
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: return Q,K
    else: return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        if num_units[0] == num_units[1]:
            outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs


def time_encoding(inputs, num_units, scope='time_kernal', reuse=None, return_weight=False):
    '''Shift-invariant time encoding kernal
    
    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: integer

    Returns:
      A 3d float tensor which embeds the input or 
      A tuple with one 3d float tensor (embeddings) and 2d float tensor (attention weight)
    '''
    assert(num_units % 2 == 0)
    effe_numits = num_units // 2
    
    with tf.variable_scope(scope, reuse=reuse):        
        init_freq_base = np.linspace(0, 8, effe_numits)
        init_freq_base = init_freq_base.astype(np.float32)
        #init_freq = 1 / 10 ** init_freq_base


        cos_freq_var = tf.get_variable('time_cos_freq', dtype=tf.float32, 
                                            initializer = tf.constant(init_freq_base))
        cos_freq_var = 1 / 10.0 ** cos_freq_var
        sin_freq_var = tf.get_variable('time_sin_freq', dtype=tf.float32, 
                                            initializer = tf.constant(init_freq_base))
        sin_freq_var = 1 / 10.0 ** sin_freq_var
        
        ones = np.ones(num_units).astype(np.float32)
        beta_var = tf.get_variable('time_beta', dtype=tf.float32, initializer = tf.constant(ones))
        
        #print(inputs.shape)
        
        expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
        #print(expand_input.shape)
        
        cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(cos_freq_var, [1, 1, effe_numits])))
        sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sin_freq_var, [1, 1, effe_numits])))
        
        freq_feat = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
        
        #print(freq_feat.shape)
        
        output = tf.multiply(freq_feat, tf.reshape(beta_var, [1, 1, num_units]))
        
        if return_weight:
            return output, tf.concat([cos_freq_var, sin_freq_var], axis=-1)
        return output
        

def basis_time_encode_flex(inputs, num_units, time_dim, expand_dim, scope='basis_time_kernal', reuse=None, return_weight=False):
    '''One version of the Mercer's time encoding

    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      time_dim: integer, number of dimention for time embedding
      expand_dim: degree of frequency expansion
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
      return_weight: bool, if true return both embeddings and frequency
    
    Returns:
      A 3d float tensor which embeds the input or 
      A tuple with one 3d float tensor (embeddings) and 2d float tensor (frequency)
    '''
    
    # inputs: [N, max_len]
    
    with tf.variable_scope('basis_time_kernal'):
        expand_input = tf.tile(tf.expand_dims(inputs, 2), [1, 1, time_dim]) # [N, max_len, time_dim]
        
        init_const = np.array([1.0 / d * np.array([1e8**(i/(time_dim-1)) * 2 * np.pi for i in range(time_dim)]) for d in range(1, expand_dim+1)]).T.astype(np.float32)
                                            
        freq_var = tf.get_variable('time_enc_freq_var', dtype=tf.float32, initializer = tf.constant(init_const))
        
        basis_expan_var = tf.get_variable('basis_expan_var', shape = [time_dim, 2*expand_dim], initializer=tf.glorot_uniform_initializer())
        
        basis_expan_var_bias = tf.get_variable('basis_expan_var_bias', shape = [time_dim], initializer=tf.zeros_initializer) #initializer=tf.glorot_uniform_initializer())

        inv_freq_var = tf.divide(tf.ones_like(freq_var), freq_var)
        sin_enc = tf.sin(tf.multiply(tf.expand_dims(expand_input,-1), tf.expand_dims(tf.expand_dims(inv_freq_var, 0),0)))
                    
        cos_enc = tf.cos(tf.multiply(tf.expand_dims(expand_input,-1), tf.expand_dims(tf.expand_dims(inv_freq_var, 0),0)))

        time_enc = tf.multiply(tf.concat([sin_enc, cos_enc], axis=-1), tf.expand_dims(tf.expand_dims(basis_expan_var,0),0))
        
        time_enc = tf.add(tf.reduce_sum(time_enc, -1), tf.expand_dims(tf.expand_dims(basis_expan_var_bias,0),0))
        
        #time_enc = tf.nn.l2_normalize(tf.add(tf.reduce_sum(time_enc, -1), tf.expand_dims(tf.expand_dims(basis_expan_var_bias,0),0)))
    if return_weight:
        return time_enc, freq_var
    return time_enc
    
    
def basis_time_encode(inputs, num_units, time_dim, expand_dim, scope='basis_time_kernal', reuse=None, return_weight=False):
    '''Mercer's time encoding

    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      time_dim: integer, number of dimention for time embedding
      expand_dim: degree of frequency expansion
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
      return_weight: bool, if true return both embeddings and frequency
    
    Returns:
      A 3d float tensor which embeds the input or 
      A tuple with one 3d float tensor (embeddings) and 2d float tensor (frequency)
    '''
    
    # inputs: [N, max_len]
    
    with tf.variable_scope('basis_time_kernal'):
        expand_input = tf.tile(tf.expand_dims(inputs, 2), [1, 1, time_dim]) # [N, max_len, time_dim]
        
        init_period_base = np.linspace(0, 8, time_dim)
        init_period_base = init_period_base.astype(np.float32)
        period_var = tf.get_variable('time_cos_freq', 
                                   dtype=tf.float32, 
                                   initializer = tf.constant(init_period_base))
        period_var = 10.0 ** period_var
        period_var = tf.tile(tf.expand_dims(period_var, 1), [1, expand_dim]) #[time_dim] -> [time_dim, 1] -> [time_dim, expand_dim]
        expand_coef = tf.cast(tf.reshape(tf.range(expand_dim) + 1, [1, -1]), tf.float32)
        
        freq_var = 1 / period_var
        freq_var = freq_var * expand_coef
        
        basis_expan_var = tf.get_variable('basis_expan_var', shape = [time_dim, 2*expand_dim], initializer=tf.glorot_uniform_initializer())
        
        basis_expan_var_bias = tf.get_variable('basis_expan_var_bias', shape = [time_dim], initializer=tf.zeros_initializer) #initializer=tf.glorot_uniform_initializer())


        sin_enc = tf.sin(tf.multiply(tf.expand_dims(expand_input,-1), tf.expand_dims(tf.expand_dims(freq_var, 0),0)))
                    
        cos_enc = tf.cos(tf.multiply(tf.expand_dims(expand_input,-1), tf.expand_dims(tf.expand_dims(freq_var, 0),0)))

        time_enc = tf.multiply(tf.concat([sin_enc, cos_enc], axis=-1), tf.expand_dims(tf.expand_dims(basis_expan_var,0),0))
        
        time_enc = tf.add(tf.reduce_sum(time_enc, -1), tf.expand_dims(tf.expand_dims(basis_expan_var_bias,0),0))

    if return_weight:
        return time_enc, freq_var
    return time_enc


def rand_time_encode(inputs, num_units, scope='rand_time_kernal', reuse=None, min_w=0, max_w=8):
    '''Bochner time encoding with uniformly random sampled frequencies

    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
      min_w: float, min(log10(period))
      max_w: float, max(log10(period))
    
    Returns:
      A 3d float tensor which embeds the input
    '''
    assert(num_units % 2 == 0)
    effe_numits = num_units // 2
    with tf.variable_scope(scope, reuse=reuse):
        sampled_freq = tf.random_uniform(
                                        [effe_numits],
                                        minval=10 ** min_w,
                                        maxval=10 ** max_w,
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None
                                    )
        sampled_freq = tf.ones_like(sampled_freq) / sampled_freq
        sampled_freq = tf.contrib.framework.sort(sampled_freq)
        expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
        cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
        sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
        
        output = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
        return output
    
def gaussian_time_encode(inputs, num_units, scope='gaussian_time_kernal', reuse=None):
    '''Bochner time encoding with frequencies sampled from Gaussian family

    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
    
    Returns:
      A 3d float tensor which embeds the input
    '''
    assert(num_units % 2 == 0)
    effe_numits = num_units // 2
    with tf.variable_scope(scope, reuse=reuse):
        sampled_freq = tf.random_normal(
                                        [effe_numits],
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None
                                    )
        
        expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
        #sampled_freq = tf.ones_like(sampled_freq) / 10 ** sampled_freq
        
        init_freq_base = np.linspace(0, 8, effe_numits) / np.pi / 2
        init_freq_base = init_freq_base.astype(np.float32)
        init_freq = 1 / 10 ** init_freq_base
        mean_vec = tf.get_variable('mean_vec', dtype=tf.float32, initializer=tf.constant(init_freq))
        std_vec = tf.ones_like(mean_vec)
        sampled_freq = std_vec * sampled_freq + mean_vec
        
        #sampled_freq = tf.contrib.framework.sort(sampled_freq)
        #print('no sort')
                        
        #print(expand_input.shape)
        
        cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
        sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
        
        output = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
        return output
    
    
def inverse_cdf_time_encode(inputs, num_units, method='maf', scope='inverse_cdf_time_kernal', reuse=None):
    '''Bochner time encoding with different inverse CDF methods

    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      method: str, method for the inverse CDF
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
    
    Returns:
      A 3d float tensor which embeds the input
    '''
    assert(num_units % 2 == 0)
    effe_numits = num_units // 2

    tfd = tfp.distributions
    tfb = tfp.bijectors
    with tf.variable_scope(scope, reuse=reuse):
        
        expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))

        if method == 'mlp_res':
            print('inv cdf method: mlp_res')
            sampled_freq = tf.random_uniform(
                                        [1, effe_numits],
                                        minval=0.0,
                                        maxval=1.0,
                                        dtype=tf.float32,
                                        seed=None,
                                        name=None
                                    )
            sampled_freq = tf.ones_like(sampled_freq) / 10 ** sampled_freq
            sampled_freq1 = tf.keras.layers.Dense(units=effe_numits, activation=tf.nn.relu, use_bias=True, bias_initializer='zeros')(sampled_freq)
            sampled_freq2 = tf.keras.layers.Dense(units=effe_numits, activation=None, use_bias=True, bias_initializer='zeros')(sampled_freq1)
            sampled_freq = tf.keras.layers.Dense(units=effe_numits, activation=None, use_bias=True, bias_initializer='zeros')(tf.add(sampled_freq2, sampled_freq))

        elif method == 'maf':
            print('inv cdf method: maf')
            maf = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0., scale=1.),
                bijector=tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=[256, 256])),
                        event_shape=[1])
            sampled_freq = maf.sample([1,effe_numits])

        elif method == 'iaf':
            print('inv cdf method: iaf')
            iaf = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0., scale=1.),
                bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=[256, 256]))),
                event_shape=[1])
            sampled_freq = iaf.sample([1,effe_numits])

        elif method == 'NVP':
            print('inv cdf method: NVP')
            nvp = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0., scale=1.),
                bijector=tfb.RealNVP(
                    num_masked=2,
                    shift_and_log_scale_fn=tfb.real_nvp_default_template(
                        hidden_layers=[256, 256])))
            sampled_freq = nvp.sample([1,effe_numits])
            
        else:
            raise ValueError('method not found')

        sampled_freq = tf.exp(sampled_freq)
        
        cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
        sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
        
        output = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
        return output
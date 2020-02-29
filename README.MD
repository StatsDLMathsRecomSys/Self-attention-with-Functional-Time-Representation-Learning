# Self-attention with Functional Time Representation Learning (NeurIPS 2019)
<!--#### -->

#### Authors: Da Xu*, Chuanwei Ruan*, Sushant Kumar, Evren Korpeoglu,  Kannan Achan
#### Please contact Da.Xu@walmartlabs.com or Chuanwei.Ruan@walmartlabs.com for questions.

#### Full paper: https://arxiv.org/abs/1911.12864

### Introduction 
Sequential modelling with self-attention has achieved cutting edge performances in natural language processing. However, like most other sequence models, self-attention does not account for the time span between events and thus captures sequential signals rather than temporal patterns. 

To bridge the gap between modelling time-independent and time-dependent event sequence, we introduce a functional feature map that embeds time span into high-dimensional spaces. By constructing the associated translation-invariant time kernel function, we reveal the functional forms of the feature map under classic functional function analysis results, namely Bochner's Theorem and Mercer's Theorem. We propose several models to learn the functional time representation and the interactions with event representation. 

These methods are evaluated on real-world datasets under various continuous-time event sequence prediction tasks. The experiments reveal that the proposed methods compare favorably to baseline models while also capturing useful time-event interactions.

![illustration](architect.png?raw=true "workflow visualization")
An illustration of a general architecture of the proposed approach. Note that we have applied different changes to the general structure on each dataset we experimented on to be consistent with the baselines. The general architecture can be easily recovered from the implementations.

### Extension to temporal graphs
In our [Inductive Representation Learning on Temporal Graphs (ICLR 2020)](https://openreview.net/pdf?id=rJeW1yHYwH) paper, we extend the methodologies proposed in the paper to temporal graph setting. Click the [link](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) to the GitHub repo.

### Datasets
Data is in the folder ``input_data``. 
* MoviesLens-1M: ``ml-1m``

* StackOverFlow: ``so``


### Code Setup
Requires: Python version >=  3.7.0 and a Linux system.

And...

```bash
pip install -r requirements.txt 
```

### Running the Experiments on Public Datasets

Run with default parameters
```bash
#movie-lens
bash exp_movieLens.sh

#stack-overflow
bash exp_so.sh
```

### Argument List

The training programs have different defaults for different dataset. The followings are the definition of the arguments used.

* `--train_dir`: name of folder to save the outcomes and logs

* `--time_basis`: use Mercer's time encoding.

* `--time_bochner`: use non-parametric Bochner's time encoding.

* `--time_rand`: Bochner's time encoding with uniformly sampled frequencies. (not mentioned in paper, for testing purpose only).

* `--time_pos`: use positional encoding instead of time encoding.

* `--time_inv_cdf`:  Bochner's inverse-CDF encoding.

* `--inv_cdf_method`: chose the method to learn the inverse CDF. Choose from [`mlp_res`, `maf`, `iaf`, `NVP`]. `mlp_res` is simple MLP based network with residual block. The rest are flow-based distributional learning methods.

* `--CUDA_device`: set the GPU to be used. 

* `--batch_sze`: batch size. 

* `--lr`: learning rate.

* `--maxlen`: maximum length of the input sequence.

* `--num_blocks`: number of attention blocks.

* `--num_epochs`: number of epochs to train the model.

* `--num_heads`: number of the heads for the multi-head attention block.

* `--dropout_rate`: probability of dropping the neuron.

* `--l2_emb`: l2 regularization on embeddings.

* `--expand_factor`: degree of expansions used for Mercer's time encoding.

* `--time_factor`: given the embedding dimension, the dimension of time embedding is determined according to (#dimension of time encoding) / (#dimension of embeddings).



### Citation

```
@inproceedings{xu2019self,
  title={Self-attention with Functional Time Representation Learning},
  author={Xu, Da and Ruan, Chuanwei and Korpeoglu, Evren and Kumar, Sushant and Achan, Kannan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15889--15899},
  year={2019}
}
```

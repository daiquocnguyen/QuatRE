<p align="center">
	<img src="https://github.com/daiquocnguyen/QuatRE/blob/master/QuatRE_logo.png">
</p>

# Relation-Aware Quaternions for Knowledge Graph Embeddings<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FQuatRE%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/QuatRE"><a href="https://github.com/daiquocnguyen/QuatRE/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/QuatRE"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/QuatRE">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/QuatRE">
<a href="https://github.com/daiquocnguyen/QuatRE/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/QuatRE"></a>
<a href="https://github.com/daiquocnguyen/QuatRE/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/QuatRE"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/QuatRE">

- This program provides the implementation of our QuatRE - a simple yet effective KG embedding model - as described in [our paper](https://arxiv.org/abs/2009.12517), where QuatRE further utilizes two relation-aware rotations to strengthen the correlations between the head and tail entities within the Quaternion space.

<p align="center">
	<img src="https://github.com/daiquocnguyen/QuatRE/blob/master/QuatRE.png" width="350">
</p>


## Usage

### News
- November 02, 2020: The extended abstract of [our paper](https://arxiv.org/abs/2009.12517) has been accepted to the NeurIPS 2020 Workshop on Differential Geometry meets Deep Learning ([DiffGeo4DL](https://sites.google.com/view/diffgeo4dl/)).


### Requirements
- Python 3.x
- Pytorch 1.5.0

### Running commands:
	
	python train_QuatRE.py --nbatches 100 --dataset WN18RR --hidden_size 256 --neg_num 5 --valid_step 400  --num_epochs 8000 --learning_rate 0.1 --lmbda 0.5 --model_name WN18RR_nb100_hs256_neg5_lr0.1_ld0.5 --mode train

	python train_QuatRE.py --nbatches 100 --dataset FB15K237 --hidden_size 384 --neg_num 10 --valid_step 200  --num_epochs 2000 --learning_rate 0.1 --lmbda 0.5 --model_name FB15K237_nb100_hs384_neg10_lr0.1_ld0.5 --mode train

	python train_QuatRE.py --nbatches 100 --dataset WN18 --hidden_size 256 --neg_num 10 --valid_step 400  --num_epochs 8000 --learning_rate 0.1 --lmbda 0.1 --model_name WN18_nb100_hs256_neg10_lr0.1_ld0.1 --mode train
	
	python train_QuatRE.py --nbatches 100 --dataset FB15K --hidden_size 384 --neg_num 5 --valid_step 200  --num_epochs 2000 --learning_rate 0.02 --lmbda 0.05 --model_name FB15K_nb100_hs384_neg5_lr0.02_ld0.05 --mode train


## Cite 

Please cite the paper whenever QuatRE is used to produce published results or incorporated into other software:

	@article{Nguyen2020QuatRE,
		author={Dai Quoc Nguyen and Thanh Vu and Tu Dinh Nguyen and Dinh Phung},
		title={QuatRE: Relation-Aware Quaternions for Knowledge Graph Embeddings},
		journal={arXiv preprint arXiv:2009.12517},
		year={2020}
	}

## License

As a free open-source implementation, QuatRE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

QuatRE is licensed under the Apache License 2.0.

The code is based on [the OpenKE framework](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)).

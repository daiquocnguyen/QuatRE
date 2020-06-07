<p align="center">
	<img src="https://github.com/daiquocnguyen/QuatRE/blob/master/QuatRE_logo.png">
</p>

# Knowledge Graph Embeddings with Quaternion Relation-Aware Attributes<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FQuatRE%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/QuatRE"><a href="https://github.com/daiquocnguyen/QuatRE/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/QuatRE"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/QuatRE">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/QuatRE">
<a href="https://github.com/daiquocnguyen/QuatRE/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/QuatRE"></a>
<a href="https://github.com/daiquocnguyen/QuatRE/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/QuatRE"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/QuatRE">

- This program provides the implementation of our KG embedding model QuatRE as described in [the paper](), where we embed entities and relations within the Quaternion space with Hamilton product.

- QuatRE utilizes the two relation-specific quaternion vectors for each relation to strengthen the correlations of the relation-aware attributes of the head and tail entities. Our proposed QuatRE obtains state-of-the-art performances on four well-known datasets including WN18, WN18RR, FB15K, and FB15k237 for the knowledge graph completion task.

<p align="center">
	<img src="https://github.com/daiquocnguyen/QuatRE/blob/master/QuatRE.png" width="350">
</p>


## Usage

### Requirements
- Python 3.x
- Pytorch 1.5.0

### Running commands:


The code is based on the OpenKE framework.

## Cite 

Please cite the paper whenever QuatRE is used to produce published results or incorporated into other software:

	@article{Nguyen2020QuatRE,
          author={Dai Quoc Nguyen and Thanh Vu and Tu Dinh Nguyen and Dinh Phung},
          title={{QuatRE: Knowledge Graph Embeddings with Quaternion Relation-Aware Attributes}},
          journal={arXiv preprint arXiv:},
          year={2020}
          }

## License

As a free open-source implementation, QuatRE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

QuatRE is licensed under the Apache License 2.0.

### ProoFVer

ProoFVer is a proof system, based on natural logic, for explainable fact verification. ProoFVer's explanations are faithful, to the decision making process of the model, as ProoFVer is faithful byconstruction. 


- ProoFVer's code repository is based on the codebase of [GENRE](https://github.com/facebookresearch/GENRE/) by [De Cao et. al.](https://github.com/facebookresearch/GENRE/graphs/contributors). We add the relevant data and scripts, along with GENRE, for training and running ProoFVer.

For training data generation:

First use the chunker, with the following script

``` scriptProofver/Chunking.py```


Then run the aligner

```scriptProofver/Alignment.py```
## Main dependencies
* python>=3.7
* pytorch>=1.6
* fairseq>=0.10
* simalign
* flair
* transformers
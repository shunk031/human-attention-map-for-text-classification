# Human Attention for Text Classification

<a href="https://www.aclweb.org/anthology/2020.acl-main.419/"><img alt="ACL2020 2020.acl-main.419" src="https://img.shields.io/badge/ACL2020-2020.acl--main.419-red"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/allenai/allennlp"><img alt="Powered by AllenNLP" src="https://img.shields.io/badge/Powered%20by-AllenNLP-blue.svg"></a>

Re-implementation of the paper [Human Attention Maps for Text Classification: Do Humans and Neural Networks Focus on the Same Words?](https://www.aclweb.org/anthology/2020.acl-main.419/) (ACL2020).

## Install requirements

```sh
$ poetry install
```

## Download and Split Yelp dataset
### Download from Yelp.com
- https://www.yelp.com/dataset/download

### Split the dataset
- The Yelp dataset is so large that it is divided into subsets in advance.
  - After that, we can get `tng.jsonl`, `val.jsonl`, and `tst.jsonl` from `data` directory.

```sh
$ allennlp split-dataset \
    --input-file data/yelp_academic_dataset_review.json \
    --output-dir data/ \
    --tng-ratio 0.8 \
    --val-ratio 0.1 \
    --tst_ratio 0.1
```

### Preprocess HAM dataset

```sh
$ allennlp preprocess-ham-dataset \
    --ham-dataset-dir data/ham-dataset/raw_data/ \
    --output-dir data/
```

## Train RNN model

```sh
$ CUDA_VISIBLE_DEVICES=0 allennlp train config/base.jsonnet -s outputs -o '{"trainer": {"cuda_device": 0}}'
```

## Reference

- Sen, Cansu, et al. "Human Attention Maps for Text Classification: Do Humans and Neural Networks Focus on the Same Words?." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.

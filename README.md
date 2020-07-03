# Human Attention for Text Classification

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/allenai/allennlp"><img alt="Powered by AllenNLP" src="https://img.shields.io/badge/Powered%20by-AllenNLP-blue.svg"></a>

## Download and Split Yelp dataset
### Download from Yelp.com
- https://www.yelp.com/dataset/download

### Split the dataset
- The Yelp dataset is so large that it is divided into subsets in advance.
  - After that, we can get `tng.jsonl`, `val.jsonl`, and `tst.jsonl' from `data` directory.

```sh
$ allennlp split-dataset \
    --input-file data/yelp_academic_dataset_review.json \
    --output-dir data/ \
    --tng-ratio 0.8 \
    --val-ratio 0.1 \
    --tst_ratio 0.1
```

## Train RNN model

```shell
$ CUDA_VISIBLE_DEVICES=0 allennlp train config/base.jsonnet -s outputs -o '{"trainer": {"cuda_device": 0}}'
```

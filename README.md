# Human Attention for Text Classification

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/allenai/allennlp"><img alt="Powered by AllenNLP" src="https://img.shields.io/badge/Powered%20by-AllenNLP-blue.svg"></a>




## Download Yelp dataset

- https://www.yelp.com/dataset/download

## Train RNN model

```shell
$ CUDA_VISIBLE_DEVICES=0 allennlp train config/base.jsonnet -s outputs -o '{"trainer": {"cuda_device": 0}}'
```

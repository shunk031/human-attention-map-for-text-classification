local seed = 19950815;

local dataset_reader = {
    "type": "yelp",
    "dataset_path": "data/yelp_academic_dataset_review.json",
};

local model = {
    "type": "rnn",
    "word_embedding": {
        "type": "basic",
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                "pretrained_file": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec"
            }
        }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 128,
        "num_layers": 1,
        "bidirectional": true
    },
    "attention": {
        "type": "tanh",
        "hidden_size": 256
    }
};

local data_loader = {
    "batch_sampler": {
        "type": "bucket",
        "batch_size" : 32,
    },
    "num_workers": 4,
};

local trainer = {
    "num_epochs": 5,
    "patience": 1,
    "validation_metric": "+acc1",
    "cuda_device": -1,
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    }
};

{
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "dataset_reader": dataset_reader,
    "train_data_path": "tng",
    "validation_data_path": "val",
    "test_data_path": "tst",
    "evaluate_on_test": true,
    "model": model,
    "data_loader": data_loader,
    "trainer": trainer,
}

# Learning to Fuse Sentences with Transformers for Summarization
Code for the EMNLP 2020 paper "[Learning to Fuse Sentences with Transformers for Summarization](https://arxiv.org/pdf/2010.03726.pdf)".

## Citation
```
@inproceedings{lebanoff-etal-2020-learning,
    title = "Learning to Fuse Sentences with Transformers for Summarization",
    author = "Lebanoff, Logan and Dernoncourt, Franck and Kim, Doo Soon and Wang, Lidan and Chang, Walter and Liu, Fei",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.338",
    doi = "10.18653/v1/2020.emnlp-main.338",
    pages = "4136--4142",
}
```

# Data
The data consists of > 100k sentence fusion examples for abstractive summarization. The sentences are taken from the CNN/DailyMail dataset. Each example is an input pair of sentences from the article, along with the output fused sentence. Each example also comes with *points of correspondence* information -- labeled spans from each sentence that indicate what connects the two sentences together (similar to coreference).

There are two versions of the data found in the `data/` directory:

1) Heuristic set -- a coarse dataset used for training the models (and for testing) 
2) Points of Correspondence set -- a high-quality dataset used only for testing

# Trained Models and Model Outputs
They will be posted here soon!

# How to train the model
Run the command to train the Trans-Linking model:
```
python run_inference.py --link --first_chain_only --first_mention_only --heuristic_dataset
```

This will place the model files in `data/output_decoding_heuristicset_link_fc_fm/`. The model with the lowest loss on the validation data will be in `data/output_decoding_heuristicset_link_fc_fm/best/`.

# How to run inference on the trained model
Run the command to run inference on a trained Trans-Linking model on the Heuristic set:
```
python run_inference.py --link --first_chain_only --first_mention_only --heuristic_dataset
```

Run the command to run inference on a trained Trans-Linking model on the Heuristic set:
```
python run_inference.py --link --first_chain_only --first_mention_only --heuristic_dataset
```

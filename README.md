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

## Presentation Video
Watch our presentation given virtually at ACL:

[![Watch our presentation given virtually at >EMNLP:](https://www.cs.ucf.edu/~feiliu/img/presentation_poc.png)](https://slideslive.com/38939343/learning-to-fuse-sentences-with-transformers-for-summarization)

# Data
The data consists of > 100k sentence fusion examples for abstractive summarization. The sentences are taken from the CNN/DailyMail dataset. Each example is an input pair of sentences from the article, along with the output fused sentence. Each example also comes with *points of correspondence* information -- labeled spans from each sentence that indicate what connects the two sentences together (similar to coreference).

There are two versions of the data found in the `data/` directory:

1) Heuristic set -- a coarse dataset used for training the models (and for testing) 
2) Points of Correspondence set -- a high-quality dataset used only for testing

# Trained Models and Model Outputs
Trans-Linking model trained on Heuristic set: https://drive.google.com/file/d/1GaOjLZ69nid_9NQiK7TcOz2TWNKXa4n8/view?usp=sharing

Baseline Transformer model trained on Heuristic set: https://drive.google.com/file/d/1oBN6nfTTre8MwVDajB8bC1uKo1hK_RYZ/view?usp=sharing

Model outputs on Heuristic set: https://drive.google.com/file/d/1K3r9t6jA8_SF32f-xL12I0yvD00wDIeh/view?usp=sharing
(Trans-Linking is in the `cnn_dm__bert_both_crdunilm_link_fc_fm_summ100.0_ninst4` directory and the baseline Transformer is in the `cnn_dm__bert_both_crdunilm_summ100.0_ninst4` directory)

Model outputs on Points of Correspondence set: https://drive.google.com/file/d/1TOA5VyyAx3BLaLSoWXL16uZVrJrjEBFc/view?usp=sharing
(Trans-Linking is in the `cnn_dm__bert_both_pocd_pocgoldunilm_link_fc_fm_summ100.0_ninst4` directory and the baseline Transformer is in the `cnn_dm__bert_both_pocd_pocgoldunilm_summ100.0_ninst4` directory)

# How to train the model
Run the command to train the Trans-Linking model:
```
python bert/run_decoding.py --do_train --link --first_chain_only --first_mention_only --heuristic_dataset
```

This will place the model files in `data/output_decoding_heuristicset_link_fc_fm/`. The model with the lowest loss on the validation data will be in `data/output_decoding_heuristicset_link_fc_fm/best/`.

# How to run inference on the trained model
Run the command to run inference on a trained Trans-Linking model on the Heuristic set:
```
python run_inference.py --link --first_chain_only --first_mention_only --heuristic_dataset
```

Run the command to run inference on a trained Trans-Linking model on the Points of Correspondence test set:
```
python run_inference.py --link --first_chain_only --first_mention_only --poc_dataset
```

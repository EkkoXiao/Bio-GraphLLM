# Bio-GraphLLM

Bio-GraphLLM is a graph-language model framework for training and evaluating drug-drug interaction and drug synergy prediction tasks.

## Required Local Files

The following files and directories are large or contain local data, so they are excluded by `.gitignore`. Place them in the repository manually before training or evaluation:

- `data/raw/`: Raw DDI and DSP data used by `synergy/combined.py` to generate combined datasets.
- `data/combined/`: Training data with `train/`, `valid/`, and `test/` splits.
- `dicts/df_rma_landm.tsv`: Cell-line gene expression data.
- `modelscope/galactica-1.3b/`: Galactica/OPT language model weights.
- `bert_pretrained/` or `distilbert_pretrained/`: Q-Former/BERT initialization weights.
- `gin_pretrained/graphcl_80.pth`: Pretrained GIN graph encoder weights.

Training outputs are written to directories such as `all_checkpoints/`, `logs/`, and `result/`.

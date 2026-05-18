# Bio-GraphLLM

Bio-GraphLLM 用于药物相互作用/药物协同等任务的图-语言模型训练与评估。

## 额外需要放入仓库根目录的文件

以下内容体积较大或含本地数据，已被 `.gitignore` 排除，需要自行放到仓库中：

- `data/raw/`: 原始 DDI、DSP 数据；用于 `synergy/combined.py` 生成混合数据。
- `data/combined/`: 训练数据，包含 `train/`、`valid/`、`test/`。
- `dicts/df_rma_landm.tsv`: 细胞系基因表达数据。
- `modelscope/galactica-1.3b/`: Galactica/OPT 语言模型权重。
- `bert_pretrained/` 或 `distilbert_pretrained/`: Q-Former/BERT 初始化权重。
- `gin_pretrained/graphcl_80.pth`: GIN 图编码器预训练权重。

训练输出会写入 `all_checkpoints/`、`logs/`、`result/` 等目录。

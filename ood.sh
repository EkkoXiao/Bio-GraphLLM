export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/villa/xlx21/anaconda3/envs/MolTC/bin/python main.py \
    --root 'data/combined_ood/train/' \
    --train_root 'data/combined_ood/train/' \
    --valid_root 'data/combined_ood/valid/' \
    --test_root 'data/combined_ood/test/' \
    --devices '2,3,4,5,6' \
    --mode 'ft' \
    --filename 'ft_Bio_GraphLLM_V1.1' \
    --opt_model './modelscope/galactica-1.3b' \
    --tune_gnn \
    --no_batch_norm True \
    --prompt '[START_SMILES]{}[END_SMILES]. ' \
    --inference_batch_size 4 \
    --batch_size 4 \
    --accumulate_grad_batches 1 \
    --max_len 64 \
    --combined True \
    --combined_ddi_ood dynas \
    --combined_dsp_ood disen \
    --combined_ddi_question 'What are the side effects of these two drugs?' \
    --combined_dsp_question 'Do the two drugs exhibit synergy effects? What is their bliss synergy score in this cell line?' \
    --llm_tune lora \
    --max_epochs 50 \
    --caption_eval_epoch 5 \
    --save_every_n_epochs 5 \

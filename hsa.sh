export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py \
    --root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/' \
    --train_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/train/' \
    --valid_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/valid/' \
    --test_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/test/' \
    --devices '1,2,3,4' \
    --mode 'ft' \
    --filename "ft_SynergyScaffoldHsa_no_decor" \
    --opt_model '/DATA/DATANAS1/xlx21/modelscope/galactica-1.3b' \
    --tune_gnn \
    --prompt '[START_SMILES]{}[END_SMILES]. ' \
    --inference_batch_size 8 \
    --batch_size 8 \
    --max_len 50  \
    --cell True \
    --NAS True \
    --gamma 0.0 \
    --question 'Do the two drugs exhibit synergy effects? What is their hsa synergy score?' \
    --llm_tune lora \
    --max_epochs 10 \
    --caption_eval_epoch 2 \
    --save_every_n_epochs 2 \
    --stage2_path /villa/xlx21/Drug-Synergy-Prediction/all_checkpoints/pretrain_SynergyScaffoldHsa_no_decor/last.ckpt 
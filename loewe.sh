export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py \
    --root '/DATA/DATANAS1/xlx21/data/SizeLoewe/' \
    --train_root '/DATA/DATANAS1/xlx21/data/SizeLoewe/train/' \
    --valid_root '/DATA/DATANAS1/xlx21/data/SizeLoewe/valid/' \
    --test_root '/DATA/DATANAS1/xlx21/data/SizeLoewe/test/' \
    --devices '2,3,4' \
    --mode 'pretrain' \
    --filename "pretrain_SynergySizeLoewe" \
    --opt_model '/DATA/DATANAS1/xlx21/modelscope/galactica-1.3b' \
    --tune_gnn \
    --prompt '[START_SMILES]{}[END_SMILES]. ' \
    --inference_batch_size 8 \
    --batch_size 8 \
    --max_len 50  \
    --cell True \
    --NAS True \
    --question 'Do the two drugs exhibit synergy effects? What is their loewe synergy score?' \
    --llm_tune lora \
    --max_epochs 20 \
    --caption_eval_epoch 5 \
    --save_every_n_epochs 5 \
    --stage2_path /villa/xlx21/Drug-Synergy-Prediction/all_checkpoints/pretrain_SynergySizeLoewe_old/last.ckpt
    
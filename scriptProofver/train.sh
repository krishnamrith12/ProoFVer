TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4

PROO_PATH=model.pt
#BART_PATH=bart.large/model.pt

NAME=proofver14 


CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train /home/ak2329/rds/hpc-work/GenreDataFiles/febFiles/feb14-bin \
    --restore-file $PROO_PATH \
    --max-tokens $MAX_TOKENS \
    --save-dir models/$NAME \
    --keep-best-checkpoints 4 \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test --patience 200 \
    --find-unused-parameters;
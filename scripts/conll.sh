DATA_DIR="/ikerlariak/igarcia945/NerQA/conll03/"
BERT_DIR="bert-base-uncased"

BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=128
MAXNORM=1.0

OUTPUT_DIR="/ikerlariak/igarcia945/NerQA/bert-base-uncased"
mkdir -p $OUTPUT_DIR

cd ..
python3 trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 4 \
--gpus="4" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 2 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $MRC_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs 20 \
--span_loss_candidates "pred_and_gold" \
--weight_span $SPAN_WEIGHT \
--warmup_steps $WARMUP \
--max_length $MAXLEN \
--gradient_clip_val $MAXNORM
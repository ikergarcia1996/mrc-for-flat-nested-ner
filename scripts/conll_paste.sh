python3 trainer.py \
--data_dir "/ikerlariak/igarcia945/NerQA/conll03/" \
--bert_config_dir "bert-base-uncased" \
--max_length 128 \
--batch_size 1 \
--gpus="4," \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr 3e-5 \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 2 \
--default_root_dir "/ikerlariak/igarcia945/NerQA/bert-base-uncased" \
--mrc_dropout 0.3 \
--bert_dropout 0.1 \
--max_epochs 20 \
--span_loss_candidates "pred_and_gold" \
--weight_span 0.1 \
--warmup_steps 0 \
--gradient_clip_val 1.0 \
--flat




CUDA_VISIBLE_DEVICES=6 python train_baseline.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./datasets/esconv/train.csv \
    --validation_file ./datasets/esconv/dev.csv \
    --test_file ./datasets/esconv/test.csv \
    --input_column history \
    --target_column target \
    --max_source_length 384 \
    --pad_to_max_length \
    --input_truncation_side left \
    --max_target_length 128 \
    --num_beams 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --metric_for_best_model bert_score_f1 \
    --greater_is_better true \
    --load_best_model_at_end \
    --fp16 \
    --predict_with_generate \
    --output_dir ./train_esconv_outputs/bart_baseline1 \
    --overwrite_output_dir \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64
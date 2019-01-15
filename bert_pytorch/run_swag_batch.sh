export SWAG_DIR=tasks/MultipleChoice/swag_data

for i in 4 8 32 64
do
    CUDA_VISIBLE_DEVICES=1,2,3,5 python run_swag.py \
      --bert_model pretrained/bert-base-uncased \
      --do_train \
      --do_lower_case \
      --do_eval \
      --data_dir $SWAG_DIR/ \
      --train_batch_size $i \
      --learning_rate 2e-5 \
      --num_train_epochs 2.0 \
      --max_seq_length 80 \
      --output_dir tasks/MultipleChoice/swag_output_base_epoch_5_batch_${i}_lr_2e_5/ \
      --gradient_accumulation_steps 4
done
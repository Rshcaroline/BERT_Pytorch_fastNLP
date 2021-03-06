export SWAG_DIR=../bert_pytorch/tasks/MultipleChoice/swag_data

python run_swag.py \
  --bert_model pretrained/bert-base-uncased \
  --do_train \
  --do_lower_case \
  --do_eval \
  --data_dir $SWAG_DIR/ \
  --train_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 80 \
  --output_dir tasks/MultipleChoice/swag_output_test/ \
  --gradient_accumulation_steps 4

for i in {1..10}
do
    CUDA_VISIBLE_DEVICES=3,4 python run_swag.py \
      --bert_model pretrained/bert-base-uncased \
      --do_train \
      --do_lower_case \
      --do_eval \
      --data_dir $SWAG_DIR/ \
      --train_batch_size 16 \
      --learning_rate 2e-5 \
      --num_train_epochs $i.0 \
      --max_seq_length 80 \
      --output_dir tasks/MultipleChoice/swag_output_base_epoch_${i}_batch_16_lr_2e_5/ \
      --gradient_accumulation_steps 4
done

for i in 4 8 16 32 64
do
    CUDA_VISIBLE_DEVICES=3,4 python run_swag.py \
      --bert_model pretrained/bert-base-uncased \
      --do_train \
      --do_lower_case \
      --do_eval \
      --data_dir $SWAG_DIR/ \
      --train_batch_size $i \
      --learning_rate 2e-5 \
      --num_train_epochs 5.0 \
      --max_seq_length 80 \
      --output_dir tasks/MultipleChoice/swag_output_base_epoch_5_batch_${i}_lr_2e_5/ \
      --gradient_accumulation_steps 4
done

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
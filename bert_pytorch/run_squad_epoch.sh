export SQUAD_DIR=tasks/QuestionAnswering/squad_data/

for i in 1 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=0,4 python run_squad.py \
  --bert_model pretrained/bert-base-uncased \
  --do_train 1\
  --do_predict 1\
  --do_lower_case 1\
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 5e-5 \
  --num_train_epochs $i.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir tasks/QuestionAnswering/squad_output_base_epoch_${i}_batch_12_lr_5e-5
done
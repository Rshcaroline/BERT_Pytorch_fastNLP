export GLUE_DIR=tasks/SequenceClassification/glue_data

for i in {1..10}
do
CUDA_VISIBLE_DEVICES=0,2 python run_classifier.py \
  --task_name CoLA \
  --do_train 1 \
  --do_eval 1 \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model pretrained/bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $i.0 \
  --output_dir tasks/SequenceClassification/CoLA_output_base_epoch_${i}_batch_32_lr_2e_5
done

for i in 32 64 128 256
do
CUDA_VISIBLE_DEVICES=0,2 python run_classifier.py \
  --task_name CoLA \
  --do_train 1 \
  --do_eval 1 \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model pretrained/bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size $i \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir tasks/SequenceClassification/CoLA_output_base_epoch_5_batch_${i}_lr_2e_5
done

for i in {1..10}
do
CUDA_VISIBLE_DEVICES=0,2,3,4 python run_classifier.py \
  --task_name CoLA \
  --do_train 1 \
  --do_eval 1 \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model pretrained/bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $i.0 \
  --output_dir tasks/SequenceClassification/CoLA_output_large_epoch_${i}_batch_32_lr_2e_5
done

for i in 32 64 128 256
do
CUDA_VISIBLE_DEVICES=0,2,3,4 python run_classifier.py \
  --task_name CoLA \
  --do_train 1 \
  --do_eval 1 \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model pretrained/bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size $i \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir tasks/SequenceClassification/CoLA_output_large_epoch_5_batch_${i}_lr_2e_5
done


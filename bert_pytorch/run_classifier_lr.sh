export GLUE_DIR=tasks/SequenceClassification/glue_data

for i in 2e-5 3e-5 4e-5 5e-5
do
CUDA_VISIBLE_DEVICES=0,2,3,4 python run_classifier.py \
  --task_name CoLA \
  --do_train 1 \
  --do_eval 1 \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model pretrained/bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate $i \
  --num_train_epochs 5.0 \
  --output_dir tasks/SequenceClassification/CoLA_output_base_epoch_5_batch_16_lr_${i}
done

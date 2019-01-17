# BERT_Pytorch_fastNLP
A PyTorch &amp; fastNLP implementation of Google AI's BERT model. 

-  Stable Version: The folder of `bert_pytorch` is the stable version of BERT, where we organized the codes based on [Pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) as the same code framework as  [fastNLP](https://github.com/fastnlp/fastNLP). 
-  Developing Version:   The folder of `bert_fastNLP` is our developing version of BERT, where we implemented our BERT model on fastNLP, the code is concise and we can use converting script to access pre-trained parameters for these implementations. In this version, we realized three specific BERT models for different tasks.

## Environment:

python >= 3.5

pytorch == 1.0

## Dataset:

### GLUE Datasets

The General Language Understanding Evaluation (GLUE) benchmark is a collection of diversenatural language understanding tasks. Most of the GLUE datasets have already existed for a numberof years, but the purpose of GLUE is to:

-  Distribute these datasets with canonical Train, Dev and Test splits.
-  Set up an evaluation server to mitigate issues with evaluation inconsistencies and Test set overfitting.

**MRPC:** Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extractedfrom online news sources with human annotations for whether the sentences in the pair are semanti-cally equivalent.

**CoLA:** The Corpus of Linguistic Acceptability is a binary single-sentence classification task, wherethe goal is to predict whether an English sentence is linguistically “acceptable” or not.

### SWAG Datasets

The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded common-sense inference.

### SQuAD v1.1 Datasets

The Stanford Question Answering Dataset (SQuAD) is a collection of 100k crowdsourced question/answer pairs. Given a question and a paragraph from Wikipedia containing the answer, thetask is to predict the answer text span in the paragraph.

## BERT-PyTorch

This version is based on [Pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) , but we organized the code as the same framework as  [fastNLP](https://github.com/fastnlp/fastNLP). 

### Quick Use:

1. Download *GLUE* Dataset to `tasks/SequenceClassification/`

2. Download Pre-trained Parameters of BERT

3. Use this command 

   ```shell
   export GLUE_DIR=tasks/SequenceClassification/glue_data
   python run_classifier.py \
     --task_name MRPC \
     --do_train 1 \
     --do_eval 1 \
     --do_lower_case\
     --data_dir $GLUE_DIR/MRPC/ \
     --bert_model pretrained/bert-base-uncased \
     --max_seq_length 128 \
     --train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3.0 \
     --output_dir tasks/SequenceClassification/mrpc_output
   ```

### How to Get Pre-trained Parameters:
Parameters from [Pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT):

| MODEL                          | LINK                                     |
| ------------------------------ | ---------------------------------------- |
| bert-base-uncased              | https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz |
| bert-large-uncased             | https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz |
| bert-base-cased                | https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz |
| bert-large-cased               | https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz |
| bert-base-multilingual-uncased | https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz |
| bert-base-chinese              | https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz |

## BERT-fastNLP

### Quick Use:

1. Download *GLUE* Dataset to `tasks/SequenceClassification/`

2. Download Pre-trained Parameters of BERT

3. Convert this Parameters into our format in `converted/`

4. Use this command 

   ```shell
   export GLUE_DIR=../bert_pytorch/tasks/SequenceClassification/glue_data
   python run_classifier_fastNLP.py \
     --task_name MRPC \
     --do_train 1 \
     --do_eval 1 \
     --do_lower_case\
     --data_dir $GLUE_DIR/MRPC/ \
     --bert_model pretrained/bert-base-uncased \
     --max_seq_length 128 \
     --train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3.0 \
     --output_dir tasks/SequenceClassification/mrpc_output
   ```

### How to Convert Pre-trained Parameters:

1. Use our  `converted/convert.py`  to converted parameters in `bert_pytorch` for our model implementation. For example, if we want to converted BERT-LARGE `pytorch_model.bin`, open this script and set:

   ```python
   ORGINAL_PATH = "../../bert_pytorch/pretrained/bert-large-uncased/pytorch_model.bin"
   OUTPUT_PATH = "large-uncased/"
   LAYERS = 24
   ```

2. For BERT-BASE, add `bert_config.json` as:

   ```json
   { "hidden": 768,
     "n_layers": 12,
     "attn_heads": 12,
     "dropout": 0.1  }
   ```

   For BERT-LARGE, add `bert_config.json` as:

   ```json
   { "hidden": 1024,
     "n_layers": 24,
     "attn_heads": 16,
     "dropout": 0.1  }
   ```

3. Copy the `vocab.txt`  from the original folder to this folder. 

### How to Use fastNLP in BERT Training:

Taking this `run_classifier_fastNLP.py` as example, where we will fine-tune BERT for classification based on  MRPC dataset. 

1. Load dataset based on fastNLP:

   ```python
   from preprocessing.sequence_classification import load_dataset

   ###### fastNLP.DataSet loading ######
   train_data, dev_data = load_dataset(args)
   ```

   where `load_dataset` will return training data and delopment data with the `fastNLP.DataSet` data type, you can find the details in `precrocess/sequence_classification`:

   ```python
   # training dataset
   train_features = convert_examples_to_features(
       train_examples, label_list, args.max_seq_length, tokenizer)
   train_data = DataSet(
       {
           "x": [f.input_ids for f in train_features],
           "segment_info": [f.segment_ids for f in train_features],
           "mask": [f.input_mask for f in train_features],
           "target": [f.label_id for f in train_features]
       }
   )
   train_data.set_input('x', 'segment_info', 'mask')
   train_data.set_target('target')
   ```

2. Build BERT-encoder model for differenet tasks, where we defined these specific model in `bert.py`, now we have implememented four models:

   ```python
   class BertMLM(backbone.Bert):
       """
       BERT Mask Language Model: Bert based model for novel task of mask language model.
       """
       
   class BertMC(backbone.Bert):
       """
       BERT Multiple Choice Model: Bert based classification model for multiple choice
       """
       
   class BertQA(backbone.Bert):
       """
       BERT Question Answering Model: Bert based model for question answering
       """
       
   class BertSC(backbone.Bert):
       """
       BERT Sequence Classification Model: Bert based classification model for sequence
       """
   ```

    In  `main()` function, we can build our model as:

   ```python
   from bert import BertMLM, BertSC, BertQA, BertMC

   model = BertSC(args.vocab_size, num_labels=args.num_labels)
   ```

   and load converted pre-trained parameters

   ```python
   MODEL_NAME = "pytorch_model.bin"
   args.bert_dir = "converted/base-uncased"

   model.load(os.path.join(args.bert_dir, MODEL_NAME))
   ```

3. Build your Optimizer, where we reuse the `BertAdam` (with `warmup`):

   ```python
   from optimization import BertAdam

   ###### ptimizer initializing ######
   optimizer = BertAdam(
       optimizer_grouped_parameters,
       lr=args.learning_rate,
       warmup=args.warmup_proportion,
       t_total=t_total
   )
   ```

4. Use fastNLP.Trainer to fine-tune the specific bert model:

   ```python
   from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric

   ###### fastNLP.Trainer initializing ######
   trainer = Trainer(model=model,
                     train_data=train_data,
                     dev_data=dev_data,
                     loss=CrossEntropyLoss(pred="pred", target="target"),
                     metrics=AccuracyMetric(),
                     print_every=1,
                     optimizer=optimizer,
                     batch_size=args.train_batch_size,
                     n_epochs=args.num_train_epochs)

   # train our model
   trainer.train()
   ```

**NOTICE:** Due to the API of fastNLP, some training tricks is difficult to be implemented directly based on `fastNLP.Trainer`. In this project, to reproduce the training and evaluation protocol, I remained the original trainining code for SWAG and SQUAD v1.1 task in `run_swag_fastNLP.py` and `run_squad_fastNLP.py` for these reasons:

-  For SWAG,  `fastNLP.Batch` will raise errors when build batches, which is corresponding to the shape of training data.
-  In SWAG training, we set `args.gradient_accumulation_steps = 4` , it seems not easy to realize gradient accumulation in `fastNLP.Trainer`. 
-  For SQUAD, we set loss function as `(CE(start, start_)+CE(end, end_))/2`. In fastNLP, it might be hard to use multiple loss functions in one training epoch.

Though we remain the orignal training code for these two tasks, we replace the orginal bert model with our version. Please check the details in `run_swag_fastNLP.py` and `run_squad_fastNLP.py` . 

### The Implementation of BERT

1. We re-implemented the `multi-head attention` and `transformer` class based on the project of [Pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT), [BERT-pytorch](https://github.com/codertimo/BERT-pytorch) and [Google-bert](https://github.com/google-research/bert).

   1. In `fastNLP.module.aggregator.attention`, our multi-head attention version is as below. It's worth noting that we found that all of these implementations above are concatentating attentions in `mutli-head attention` weight-wisely which is more user-friendly and efficient. Therefore we don't apply the existing basic `Attention` class in fastNLP to implement `MultiHeadAtte`.

      ```python
      class MultiHeadAtte(nn.Module):
          def __init__(self, input_size, output_size, hidden_size, num_atte, dropout):
              super(MultiHeadAtte, self).__init__()
              self.num_attention_heads = num_atte
              self.attention_head_size = int(hidden_size / self.num_attention_heads)
              self.all_head_size = self.num_attention_heads * self.attention_head_size

              self.query = nn.Linear(hidden_size, self.all_head_size)
              self.key = nn.Linear(hidden_size, self.all_head_size)
              self.value = nn.Linear(hidden_size, self.all_head_size)

              self.dropout = nn.Dropout(dropout)

              self.dense = nn.Linear(hidden_size, hidden_size)
              self.LayerNorm = LayerNormalization(hidden_size, eps=1e-12)
      ```

   2.  In `fastNLP.module.encoder.transformer`, we implemented our `TransformerEncoder` based on `SubLayer` with `MultiHeadAtte`. We can pay attention to that:

         ```python
         class TransformerEncoder(nn.Module):
            def __init__(self, num_layers, **kargs):
                super(TransformerEncoder, self).__init__()
                self.layers = nn.ModuleList([self.SubLayer(**kargs) for _ in range(num_layers)])
         ```

      For `self.layers`, we used `nn.ModuleList` instead of `nn.Sequential` considering that in some task, outputs of all layers are valuable. Because of this, we set flag `all_output=True`.

2. We implemented `Bert` class in `backbone.py`, where we regared `Bert` like backbone models (eg. ResNet50) in Computer Vision. We implemented the backbone model here:

   ```python
   class Bert(nn.Module):
       """
       BERT model : Bidirectional Encoder Representations from Transformers.
       """
       def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
           super().__init__()
           self.hidden = hidden
           self.n_layers = n_layers
           self.attn_heads = attn_heads
           # paper noted they used 4*hidden_size for ff_network_hidden_size
           self.feed_forward_hidden = hidden * 4
           # embedding for BERT, sum of positional, segment, token embeddings
           self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
           # multi-layers transformer blocks, deep network
           self.transformer = Transformer(
               num_layers=n_layers, 
               num_atte=attn_heads,
               input_size=hidden,
               intermediate_size=self.feed_forward_hidden,
               key_size=hidden,
               output_size=hidden,
               activate=GeLU,
               dropout=dropout,
           )
           # Pooling layer
           self.pooler = nn.Linear(hidden, hidden)
           self.activation = nn.Tanh()
   ```

3. Task-specific Bert Models are all defined in `bert.py`, we inherit the `backbone.Bert` and add the decoder part very simply. Besides of this, it's easy to load model pre-trained parameters use `model.load()` which is implemented in `backbone.Bert`. Taking `BertQA` as example, where the BertEncoder part is undertaken by `self.bert_forward`.

   ```python
   class BertQA(backbone.Bert):
       """
       BERT Question Answering Model: Bert based classification model for question answering
       """
       def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
           """
           :param vocab_size: vocab_size of total words
           :param hidden: BERT model hidden size
           :param n_layers: numbers of Transformer blocks(layers)
           :param attn_heads: number of attention heads
           :param dropout: dropout rate
           """
           super(BertQA, self).__init__(vocab_size, hidden, n_layers, attn_heads, dropout)
           self.qa_classifier = nn.Linear(hidden, 2)

       def forward(self, x, segment_info=None, mask=None):
           output_layer, _ = self.bert_forward(x, segment_info, mask=mask, all_output=False)
           start, end = self.qa_classifier(output_layer).split(1, dim=-1)
           return {'pred_start': start.squeeze(-1), 'pred_end': end.squeeze(-1)}
   ```



## Contributor:

Shihan Ran ([RshCaroline](https://github.com/Rshcaroline))

Zhankui He ([AaronHeee](https://github.com/AaronHeee))



## RelatedRepo:

-  [Google-bert](https://github.com/google-research/bert)
-  [Pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
-  [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)


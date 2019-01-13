import torch

ORGINAL_PATH = "/Users/aaronhe/Documents/NutStore/Aaron He/FDU/05 - Neural Network and Deep Learning/BERT_Pytorch_fastNLP/bert_pytorch/pretrained/bert-base-uncased/pytorch_model.bin"
LAYERS = 12

original_state_dict = torch.load(ORGINAL_PATH)

convert_state_dict = {}

convert_state_dict['embedding.token.embed.weight'] = original_state_dict['bert.embeddings.word_embeddings.weight']
convert_state_dict['embedding.position.embed.weight'] = original_state_dict['bert.embeddings.position_embeddings.weight']
convert_state_dict['embedding.segment.embed.weight'] = original_state_dict['bert.embeddings.token_type_embeddings.weight']
convert_state_dict['embedding.LayerNorm.a_2'] = original_state_dict['bert.embeddings.LayerNorm.gamma']
convert_state_dict['embedding.LayerNorm.b_2'] = original_state_dict['bert.embeddings.LayerNorm.beta']

for i in range(LAYERS):
    convert_state_dict['transformer.layers.%d.atte.query.weight' % i] = original_state_dict['bert.encoder.layer.%d.attention.self.query.weight' % i]
    convert_state_dict['transformer.layers.%d.atte.query.bias' % i] = original_state_dict['bert.encoder.layer.%d.attention.self.query.bias' % i]
    convert_state_dict['transformer.layers.%d.atte.key.weight' % i] = original_state_dict['bert.encoder.layer.%i.attention.self.key.weight' % i]
    convert_state_dict['transformer.layers.%d.atte.key.bias' % i] = original_state_dict['bert.encoder.layer.%d.attention.self.key.bias' % i]
    convert_state_dict['transformer.layers.%d.atte.value.weight' % i] = original_state_dict['bert.encoder.layer.%d.attention.self.value.weight' % i]
    convert_state_dict['transformer.layers.%d.atte.value.bias' % i] = original_state_dict['bert.encoder.layer.%d.attention.self.value.bias' % i]    
    convert_state_dict['transformer.layers.%d.atte.dense.weight' % i] = original_state_dict['bert.encoder.layer.%d.attention.output.dense.weight' % i]
    convert_state_dict['transformer.layers.%d.atte.dense.bias' % i] = original_state_dict['bert.encoder.layer.%d.attention.output.dense.bias' % i]
    convert_state_dict['transformer.layers.%d.atte.LayerNorm.a_2' % i] = original_state_dict['bert.encoder.layer.%d.attention.output.LayerNorm.gamma' % i]
    convert_state_dict['transformer.layers.%d.atte.LayerNorm.b_2' % i] = original_state_dict['bert.encoder.layer.%d.attention.output.LayerNorm.beta' % i]
    convert_state_dict['transformer.layers.%d.intermediate.0.weight' % i] = original_state_dict['bert.encoder.layer.%d.intermediate.dense.weight' % i]
    convert_state_dict['transformer.layers.%d.intermediate.0.bias' % i] = original_state_dict['bert.encoder.layer.%d.intermediate.dense.bias' % i]
    convert_state_dict['transformer.layers.%d.output.0.weight' % i] = original_state_dict['bert.encoder.layer.%d.output.dense.weight' % i]
    convert_state_dict['transformer.layers.%d.output.0.bias' % i] = original_state_dict['bert.encoder.layer.%d.output.dense.bias' % i]
    convert_state_dict['transformer.layers.%d.LayerNorm.a_2' % i] = original_state_dict['bert.encoder.layer.%d.output.LayerNorm.gamma' % i]
    convert_state_dict['transformer.layers.%d.LayerNorm.b_2' % i] = original_state_dict['bert.encoder.layer.%d.output.LayerNorm.beta' % i]

convert_state_dict['pooler.weight'] = original_state_dict['bert.pooler.dense.weight']
convert_state_dict['pooler.bias'] = original_state_dict['bert.pooler.dense.bias']

torch.save(convert_state_dict, "converted/" + ORGINAL_PATH.split("/")[-1])
import tensorflow as tf
#import tensorflow_addons as tfa

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

#test = None
# can be up to 512 for BERT
MAX_LENGTH = 256
BATCH_SIZE = 1

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case = False)

#######################INICIOS FUNÇÕES DE APOIO#######################

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def convert_example_to_feature(titulo_1, titulo_2):
    return tokenizer.encode_plus(titulo_1, titulo_2,
                                 add_special_tokens = True, # adiciona [CLS], [SEP]
                                 max_length = MAX_LENGTH, # comprimento máximo do texto de entrada
                                 padding = 'max_length', # adiciona [PAD] até o tam_max (MAX_LENGTH)
                                 truncation = True, # padrão = 'longest_first'
                                 return_attention_mask = True, # adiciona máscara de atenção para não focar nos tokens do pad
                                )

def encode_examples(df_titulos, labels, limit = -1):
    
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    
    if (limit > 0):
        ds = ds.take(limit)
    
    # for review, label in tfds.as_numpy(ds):
    for titulo_1, titulo_2, label in zip(df_titulos["titulo_1"], df_titulos["titulo_2"], labels):
        
        bert_input = convert_example_to_feature(titulo_1, titulo_2)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
        
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def formatar_dados_bert(X_treino, y_treino, X_val, y_val, X_teste, y_teste):
    
    # dataset de treino
    ds_treino = encode_examples(X_treino, y_treino).batch(BATCH_SIZE)

    # dataset de validação
    ds_val = encode_examples(X_val, y_val).batch(BATCH_SIZE)

    # dataset de teste
    ds_teste = encode_examples(X_teste, y_teste).batch(BATCH_SIZE)

    return ds_treino, ds_val, ds_teste



def fazer_predicao_teste(modelo, ds_teste):

    # Predição no dataset de teste
    tf_saida = modelo.predict(ds_teste)[0]
    tf_predicao = tf.nn.softmax(tf_saida, axis = 1)

    pred = tf.argmax(tf_predicao, axis = 1)

    y_pred = pred.numpy()

    return y_pred


#######################FIM FUNÇOES DE APOIO#######################
def pipeline_bert(X_treino, y_treino, X_val, y_val, X_teste, y_teste):

    lr = 2e-5
    num_epocas = 3

    ds_treino, ds_val, ds_teste = formatar_dados_bert(X_treino, y_treino, X_val, y_val, X_teste, y_teste)
    
    # inicialização do mocelo
    modelo = TFBertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', from_pt = True)

    # escolhendo o otimizador
    otimizador = Adam(learning_rate = lr, epsilon = 1e-08)
    perda = SparseCategoricalCrossentropy(from_logits = True)
    metrica = SparseCategoricalAccuracy('accuracy')
    modelo.compile(optimizer = otimizador, loss = perda, metrics = [metrica])

    # Fine-tune do modelo
    bert_historico = modelo.fit(ds_treino, epochs = num_epocas, validation_data = ds_val)
    
    # Predizendo a saída dos dados de teste
    y_pred = fazer_predicao_teste(modelo, ds_teste)

    return (modelo, bert_historico, y_pred)

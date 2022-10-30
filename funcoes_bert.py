import torch
import tensorflow as tf
#import tensorflow_addons as tfa

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from transformers import TFBertForSequenceClassification
from transformers import BertModel, TFBertModel
from transformers import BertTokenizer

from scipy import spatial

#test = None
# can be up to 512 for BERT
MAX_LENGTH = 256
BATCH_SIZE = 1

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case = False)


#########################################################################################################
##################################### BERT RETORNANDO CLASSIFICAÇÃO #####################################
#########################################################################################################
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


def pipeline_bert_clas(X_treino, y_treino, X_val, y_val, X_teste, y_teste):

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



#########################################################################################################
####################################### BERT RETORNANDO EMBEDDING #######################################
#########################################################################################################
def mostrar_status(BERT_hidden_states, camada_i = 0, batch_i = 0, token_i = 0):

    print( f"Número de Camadas: { len(BERT_hidden_states) }\t(Camada inicial + 12 camadas BERT)")
    print( f"Número de Batches: { len(BERT_hidden_states[camada_i]) }" )
    print( f"Número de Tokens: { len(BERT_hidden_states[camada_i][batch_i]) }" )
    print( f"Número de Hidden Units: { len(BERT_hidden_states[camada_i][batch_i][token_i]) }" )


def adicionar_token(texto):

    texto_token = "[CLS] " + texto + " [SEP]"
    return texto_token


def fazer_tokenizacao(texto):

    texto_tokenizado = tokenizer.tokenize(texto)
    return texto_tokenizado


def formatar_entrada_bert(texto):

    texto_token = adicionar_token(texto)
    texto_tokenizado = fazer_tokenizacao(texto_token)

    return texto_tokenizado


def calcular_media_2_ate_ultima_camada(BERT_hidden_states):

    vetor_token = BERT_hidden_states[-2][0] # pega da segunda camada até a ultima
    #print(f"Tamanho do vetor_token: {vetor_token.shape}")

    embedding = torch.mean(vetor_token, dim = 0) # calcula a média dos vetores de token
    #print(f"A frase tem um vetor de tamanho: {embedding.size()}")

    return embedding


def calcular_distancia(modelo, doc1, doc2):
    
    data = [doc1, doc2]
    sentence_embeddings = modelo.encode(data)

    infer1 = sentence_embeddings[0]
    infer2 = sentence_embeddings[1]
    
    cos_similarity = 1 - spatial.distance.cosine(infer1, infer2) #de 0 a 1
    
    return cos_similarity


def calcular_embedding(modelo, texto):

    texto_tokenizado = formatar_entrada_bert(texto)

    indices_token = tokenizer.convert_tokens_to_ids(texto_tokenizado) # mapeia os indices dos tokens

    segments_ids = [1] * len(texto_tokenizado) # marca todos os textos para pertecerem a sentença "1"

    tokens_tensor = torch.tensor([indices_token]) # input ids
    segments_tensors = torch.tensor([segments_ids]) # attention mask

    with torch.no_grad(): # roda o bert e salva o resultado
        saidas_bert = modelo(tokens_tensor, segments_tensors)
        #saidas.keys() # (last_hidden_state, pooler_output, hidden_states)

    BERT_hidden_states = saidas_bert[2] # 0 é last_hidden_state; 1 é pooler_output; 2 é hidden_states

    #mostrar_status(BERT_hidden_states)

    embedding = calcular_media_2_ate_ultima_camada(BERT_hidden_states)

    return embedding


def pipeline_bert(X_teste):
    
    # inicialização do mocelo
    modelo = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased', output_hidden_states = True)
 
    modelo.eval()

    #embedding = X_teste.apply(lambda linha: calcular_embedding(modelo, linha['titulo']) , axis = 1)
    embedding = X_teste.apply( lambda linha: calcular_embedding(modelo, linha) )
    
    return embedding
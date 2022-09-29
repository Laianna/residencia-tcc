import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

test = None
# can be up to 512 for BERT
MAX_LENGTH = 256
BATCH_SIZE = 64

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

def get_bert_data(X_train, y_train, X_valid, y_valid, X_test, y_test):
    
    # train dataset
    ds_train = encode_examples(X_train, y_train).batch(BATCH_SIZE)

    # test dataset
    ds_test = encode_examples(X_test, y_test).batch(BATCH_SIZE)

    #validation dataset
    ds_valid = encode_examples(X_valid, y_valid).batch(BATCH_SIZE)

    return ds_train, ds_valid, ds_test

#######################FIM FUNÇOES DE APOIO#######################

def get_test_prediction(model, ds_test):

    #Predictin test dataset
    tf_output = model.predict(ds_test)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis = 1)
    label = tf.argmax(tf_prediction, axis = 1)
    label_pred = label.numpy()
    # print(label_pred)

    return label_pred


def pipeline_bert(name, X_train, y_train, X_valid, y_valid, X_test, y_test): #X_train = [titulos1, titulos2]

    learning_rate = 2e-5
    number_of_epochs = 3
    ds_train, ds_valid, ds_test = get_bert_data(X_train, y_train, X_valid, y_valid, X_test, y_test)
    
    # model initialization
    model = TFBertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', from_pt = True)

    # choosing Adam optimizer
    optimizer = Adam(learning_rate=learning_rate, epsilon = 1e-08)
    loss = SparseCategoricalCrossentropy(from_logits = True)
    metric_acc = SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer = optimizer, loss = loss, metrics = [metric_acc])

    #Training model
    bert_history = model.fit(ds_train, epochs = number_of_epochs, validation_data = ds_valid)
    
    #Predict test data
    label_pred = get_test_prediction(model, ds_test)

    return (name, bert_history, y_test, label_pred)

    # metrics = calc_metrics(y_test, result, name)
    # return metrics
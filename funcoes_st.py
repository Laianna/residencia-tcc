#Bibliotecas
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers import evaluation



# Constantes
NOME_MODELO = (
               'distiluse-base-multilingual-cased-v1',
               'distiluse-base-multilingual-cased-v2',
               'paraphrase-multilingual-MiniLM-L12-v2',
               'paraphrase-multilingual-mpnet-base-v2'
              )

APELIDO_MODELO = (
                  'dbm-v1',
                  'dbm-v2',
                  'pml-v2',
                  'pmb-v2'
                 )



def setar_modelo(numero_modelo):

    nome_modelo = NOME_MODELO[numero_modelo]
    apelido_modelo = APELIDO_MODELO[numero_modelo]

    return nome_modelo, apelido_modelo


def retornar_input_example(titulo_1, titulo_2, label):

    return(  InputExample(texts = [ titulo_1, titulo_2 ], label = label)  )


def calcular_embedding(modelo, titulo):

    embedding = modelo.encode(titulo)

    return embedding


def pipeline_st(df_treino, df_val, df_teste, nome_modelo, num_epocas = 3):
    
    # carregando o modelo
    modelo = SentenceTransformer(nome_modelo)

    dados_treino = df_treino.apply(lambda row: retornar_input_example(row['titulo_1'], row['titulo_2'], float(row['match'])), axis = 1)
    dados_val = df_val.apply(lambda row: retornar_input_example(row['titulo_1'], row['titulo_2'], float(row['match'])), axis = 1)

    treino_dataloader = DataLoader(dados_treino, shuffle = True, batch_size = 1)
    treino_perda = losses.CosineSimilarityLoss(model = modelo)
    #treino_perda = losses.ContrastiveLoss(model = modelo)
    avaliador = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dados_val)
    #avaliador = evaluation.BinaryClassificationEvaluator.from_input_examples(dados_val)

    # fine-tune do modelo
    modelo.fit(
               train_objectives = [(treino_dataloader, treino_perda)],
               epochs = num_epocas,
               evaluator = avaliador
              )
    
    # calculando os embeddings
    embedding = df_teste.apply(lambda row: calcular_embedding(modelo, row['titulo']), axis = 1)

    return embedding
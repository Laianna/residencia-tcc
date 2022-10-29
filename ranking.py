import pandas as pd
import numpy as np
import string
import unicodedata

from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer

def remove_pontuacao(texto):
    
    texto_sp = texto.translate(str.maketrans('', '', string.punctuation))
    
    return texto_sp


def tirar_acento(texto):

    texto_sa = unicodedata.normalize("NFD", texto)
    texto_sa = texto_sa.encode("ascii", "ignore")
    texto_sa = texto_sa.decode("utf-8")

    return texto_sa


def fazer_pre_processamento(df_teste):

    # removendo a pontuação do título
    df_teste["titulo_sp_1"] = df_teste["titulo_cb_1"].apply(lambda x: remove_pontuacao(x))
    df_teste["titulo_sp_2"] = df_teste["titulo_cb_2"].apply(lambda x: remove_pontuacao(x))

    # removendo acentos do título
    df_teste["titulo_sa_1"] = df_teste["titulo_sp_1"].apply(lambda x: tirar_acento(x))
    df_teste["titulo_sa_2"] = df_teste["titulo_sp_2"].apply(lambda x: tirar_acento(x))


def concatenar_titulos(df_teste):

    lista_df = [df_teste['titulo_sa_1'], df_teste['titulo_sa_2']]
    df = pd.concat(lista_df, ignore_index = True)

    return df


def concatenar_df(df_teste, coluna_1 = "titulo_sa_1", coluna_2 = "titulo_sa_2", coluna_saida = 'titulo_sa'):

    df_1 = df_teste.rename(columns = {coluna_1: coluna_saida, 'ean_1': 'ean'})
    df_2 = df_teste.rename(columns = {coluna_2: coluna_saida, 'ean_2': 'ean'})

    lista_df = [df_1[[coluna_saida, 'ean', 'categoria']], df_2[[coluna_saida, 'ean', 'categoria']]]
    df = pd.concat(lista_df, ignore_index = True)

    return df


def calcular_tam_max(df):

    '''results_1 = set()
    df["titulo_sa_1"].str.lower().str.split().apply(results_1.update)
    tam_max_1 = len(results_1)

    results_2 = set()
    df["titulo_sa_2"].str.lower().str.split().apply(results_2.update)
    tam_max_2 = len(results_2)

    tam_max = tam_max_2 if tam_max_1 >= tam_max_2 else tam_max_1'''

    results = set()
    df.str.lower().str.split().apply(results.update)
    tam_max = len(results)

    return tam_max


def criar_dicionario(indice, titulo, ean, categoria, colunas):
    
    return {
            colunas[0] : indice, colunas[1] : titulo, colunas[2] : ean, colunas[3] : categoria
           }


def criar_df_match(df_concat, ean_repetido, colunas, coluna_saida = 'titulo_sa'):

    df_matches = pd.DataFrame(columns = colunas)
    for ean in ean_repetido:

        # pega o indice da primeira linha com aquele EAN
        filtro = (df_concat['ean'] == ean)
        indice = next(iter(filtro.index[filtro]))

        dicionario = criar_dicionario(
                                      indice = indice,
                                      titulo = df_concat.loc[indice][coluna_saida],
                                      ean = df_concat.loc[indice]['ean'],
                                      categoria = df_concat.loc[indice]['categoria'],
                                      colunas = colunas
                                     )

        df_matches = df_matches.append(dicionario, ignore_index = True)

    df_matches.sort_values('indice', inplace = True)
    df_matches.reset_index(drop = True, inplace = True)

    return df_matches


def criar_df_teste(df_teste, coluna_1 = "titulo_sa_1", coluna_2 = "titulo_sa_2", coluna_saida = 'titulo_sa'):

    COLUNAS = ("indice", coluna_saida, "ean", "categoria")

    # colocando os titulos em um dataframe com 1 coluna só
    df_concat = concatenar_df(df_teste, coluna_1, coluna_2, coluna_saida)

    # encontrando a lista de valores únicos de EAN repetidos
    vc = df_concat['ean'].value_counts()
    ean_repetido = vc[vc > 1].index.values

    # criando um dataframe apenas com 1 ocorrência de cada EAN repetido
    df_matches = criar_df_match(df_concat, ean_repetido, COLUNAS, coluna_saida)

    return df_concat, df_matches


def formatar_entrada_bow(dados, mf = 1000):
    
    cv = CountVectorizer(
                         lowercase = True,
                         strip_accents = 'unicode',
                         max_features = mf
                        )

    cv.fit(dados)
    dados_transformados = cv.transform(dados).toarray()

    #X = matriz.fit_transform(dados).toarray()
    
    return cv, dados_transformados


def calcular_dis_cos(vetor_1, vetor_2):
                
    return (1 - spatial.distance.cosine(vetor_1, vetor_2))


def ordenar_resultado(res):

    indices = []
    valores = []

    for i in range(len(res)):
        
        ind = res[i].argsort()
        val = res[i][res[i].argsort()]

        indices.append( list(reversed(ind)) )
        valores.append( list(reversed(val)) )

    return indices, valores


def calcular_dis_2_vetores(titulo_bow):

    resultado = np.zeros((len(titulo_bow), len(titulo_bow)))

    for i in range(len(titulo_bow)):

        for j in range(len(titulo_bow)):

            if i == j:
                resultado[i][j] = -1
            else:
                resultado[i][j] = calcular_dis_cos(titulo_bow[i], titulo_bow[j])

    return resultado


def calcular_dis_2_vetores_cond(titulo_bow, df_matches):

    resultado = np.zeros((len(titulo_bow), len(titulo_bow)))

    for i in df_matches['indice'].to_list():

        for j in range(len(titulo_bow)):

            if i == j:
                resultado[i][j] = -1
            else:
                resultado[i][j] = calcular_dis_cos(titulo_bow[i], titulo_bow[j])

    return resultado


def calcular_acuracia_k(df_matches, df_concat, indices):

    for k in [1, 10, 50]:

        # criando uma coluna nova no df
        df_matches[f'k-{k}'] = 0
        df_matches[f'k-{k}-qtd'] = 0
        df_matches[f'k-{k}-qtd/{k}'] = 0

        for enum, i in enumerate(df_matches['indice'].to_list()):

            for j in range(k):
                
                if df_concat['ean'].loc[i] == df_concat['ean'].loc[indices[i][j]]:

                    df_matches[f'k-{k}'].loc[enum] = 1
                    df_matches[f'k-{k}-qtd'].loc[enum] += 1

            df_matches[f'k-{k}-qtd/{k}'].loc[enum] = (df_matches[f'k-{k}-qtd'].loc[enum])/k


def calcular_match_rank(df_matches, df_concat, indices):

    # criando uma coluna nova no df
    df_matches['match_rank'] = 0
    df_matches['1/match_rank'] = 0
    df_matches['match_rank/total'] = 0
    tam_df = df_concat.shape[0]

    # para cada linha do dataframe
    for enum, i in enumerate(df_matches['indice'].to_list()):

        # para cada uma das distâncias encontradas rankeadas em ordem decrescente
        for cont, j in enumerate(indices[i]):
        
            # se for match
            if (df_concat['ean'].loc[i] == df_concat['ean'].loc[j]) and (i != j):

                # guarda a posição do primeiro match, lembrando que o enumerate começa em 0
                df_matches['match_rank'].loc[enum] = (cont + 1)
                df_matches['1/match_rank'].loc[enum] = 1/(cont + 1)
                df_matches['match_rank/total'].loc[enum] = (cont + 1)/tam_df
                
                # para o for
                break


def calcular_tempo(df_matches, inicio_tempo, final_tempo):

    tempo = final_tempo - inicio_tempo

    df_matches['tempo'] = tempo


def calcular_metricas(df_matches, df_concat, resultado):

    # colocando o resultado em ordem (menor distância até maior distância)
    indices, valores = ordenar_resultado(resultado)

    # calculando as métricas
    calcular_acuracia_k(df_matches, df_concat, indices)
    calcular_match_rank(df_matches, df_concat, indices)

    return indices, valores
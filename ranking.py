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


def calcular_dis_2_vetores_cond(df, titulo_bow, ean_repetido):

    resultado = np.zeros((len(titulo_bow), len(titulo_bow)))

    for i in range(len(titulo_bow)):

        if df['ean'].loc[i] in ean_repetido:

            for j in range(len(titulo_bow)):

                if i == j:
                    resultado[i][j] = -1
                else:
                    resultado[i][j] = calcular_dis_cos(titulo_bow[i], titulo_bow[j])

    return resultado
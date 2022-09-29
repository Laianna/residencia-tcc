import numpy as np
import string
import optuna

from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


# Remove a pontuação do texto de acordo com os caracters presentes em string.punctuation
# string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def remove_pontuacao(texto):
    
    #texto_sp = "".join([i for i in texto if i not in string.punctuation])
    texto_sp = texto.translate(str.maketrans('', '', string.punctuation))
    
    return texto_sp


#Tranforma o dataframe de dados em uma lista única de sentenças
def get_list_dataframe(dataframe):
    
    lista_titulos = []
    for titulo_1, titulo_2 in zip(dataframe["titulo_1"], dataframe["titulo_2"]):

        lista_titulos.append(remove_pontuacao(titulo_1))
        lista_titulos.append(remove_pontuacao(titulo_2))
        
    return lista_titulos


#Cria bow binário das sentenças
def vectorize_dataframe(dataframe, binario = True):
    lista_titulos = get_list_dataframe(dataframe)

    vectorizer = CountVectorizer(analyzer = "word",
                                tokenizer = None,
                                lowercase = True,
                                strip_accents = 'unicode',
                                binary = binario
                                ) 

    vector = vectorizer.fit_transform(lista_titulos).toarray()

    return vector, vectorizer


#Cria a matrix de coocorrencia do título1 e título2  para o dataset de treino
def get_cooccurrence_bow(dataframe):
    
    vector, vectorizer = vectorize_dataframe(dataframe)
    lista_features = []

    for indice in range(0, len(vector), 2):
        
        vec_titulo_1 = vector[indice]
        vec_titulo_2 = vector[indice + 1]
        
        lista_coo = np.multiply(np.logical_and(vec_titulo_1, vec_titulo_2), 1).tolist()
        
        lista_features.append(lista_coo)

    return lista_features, vectorizer

#Cria a matrix de coocorrencia do título1 e título2 para o dataset de teste e validação
def get_cooccurrence_bow_test_val(vectorizer_treino, X_):
    
    co_, cv_ = get_cooccurrence_bow(X_)
    
    co_vector = np.zeros((len(co_), len(vectorizer_treino.vocabulary_)), dtype = np.int64).tolist()
    
    for linha in range(len(co_)):

        for chave, dic_indice in zip(cv_.vocabulary_.keys(), cv_.vocabulary_.values()):
            
            if chave in vectorizer_treino.vocabulary_:             
                # print(linha,"-",dic_indice,"-",chave)
                co_vector[linha][dic_indice] = co_[linha][dic_indice]
        
    return co_vector


#Gera a matrix de coocorrência de todos os datasets
def get_all_bows(X_train, X_valid, X_test):

    X_train_coo, X_train_vectorizer = get_cooccurrence_bow(X_train)
    X_valid_coo = get_cooccurrence_bow_test_val(X_train_vectorizer, X_valid)
    X_test_coo = get_cooccurrence_bow_test_val(X_train_vectorizer, X_test)

    return X_train_coo, X_valid_coo, X_test_coo


#Define os melhores parâmetros do classificador
def get_best_parameters(X_train_vec, X_valid_vec, y_train, y_valid):
    study = optuna.create_study(direction="maximize")
    obj_parcial = partial(objective, X_train_vec, X_valid_vec, y_train, y_valid)
    study.optimize(obj_parcial, n_trials=20)

    return study.best_params


#Rodando as funções do optuna
def objective(X_train_vec, X_valid_vec, y_train, y_valid, trial):

    param_grid = {"n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 128, log=True),
                "criterion": trial.suggest_categorical('criterion', ["gini", "entropy"])}

    forest = RandomForestClassifier(**param_grid) 
    forest = forest.fit(X_train_vec, y_train)
    preds = forest.predict(X_valid_vec)
 
    from sklearn.metrics import f1_score
    metrica = f1_score(y_valid, preds, average=None)[0]

    #from sklearn.metrics import accuracy_score
    #metrica = accuracy_score(y_valid, preds)
    
    return metrica


def pipeline_rf(name_dataset, X_train, y_train, X_valid, y_valid, X_test, y_test):

    #Get BoW of the sentences
    X_train_vec, X_valid_vec, X_test_vec = get_all_bows(X_train, X_valid, X_test)

    # #Get best parameters
    best_params = get_best_parameters(X_train_vec, X_valid_vec, y_train, y_valid)
    
    #Train model with best parameters
    forest = RandomForestClassifier(**best_params) 
    #forest = RandomForestClassifier() 
    forest = forest.fit(X_train_vec, y_train)

    #Predict test dataset 
    label_pred = forest.predict(X_test_vec)

    return (name_dataset, y_test, label_pred, forest)
    
    # return X_train_vec, X_valid_vec, X_test_vec

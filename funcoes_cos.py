import funcoes_bow as fbow
from scipy import spatial


def get_cos(dataframe):

    cosine_list = []
    
    vectors, vectorizer = fbow.vectorize_dataframe(dataframe, binario = False)
        
    for i in range(0, len(vectors), 2):
        cosine_list.append(1 - spatial.distance.cosine(vectors[i], vectors[i+1]))
    
    return cosine_list


def cos_threshold(name_dataset, df_X, y_test, threshold):
    
    df_X["cos_sim"] = get_cos(df_X)
    df_X[f'cos{threshold}'] = (df_X["cos_sim"]>threshold).astype(int)
    y_pred = df_X[f'cos{threshold}']

    return name_dataset, y_test, y_pred
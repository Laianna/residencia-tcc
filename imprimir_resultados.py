import pandas as pd

# Constantes
#LISTA_CATEGORIAS = ("celulares", "notebooks", "geladeiras", "fogoes", "tvs")

ARQUIVOS = ('ale_1_1', 'ale_5_1', 'hn_1_1', 'hn_5_1')


def print_acu_k(df, flag = True):

    linha_1 = f"  | k-1\t | k-10\t| k-50\t|"

    if flag == True:
        linha_2 = f"1 | {df['k-1'].value_counts()[1]}\t | {df['k-10'].value_counts()[1]}\t| {df['k-50'].value_counts()[1]}\t|"
    else:
        linha_2 = f"1 | {df['k-1'].value_counts()[1]} | {df['k-10'].value_counts()[1]}\t| {df['k-50'].value_counts()[1]}\t|"
    linha_3 = f"0 | {df['k-1'].value_counts()[0]}\t | {df['k-10'].value_counts()[0]}\t| {df['k-50'].value_counts()[0]}\t|"

    print(f"{linha_1}\n{linha_2}\n{linha_3}")


def print_describe(df):

    print("")
    print(df[["k-1", "k-10", "k-50", 'match_rank', '1/match_rank']].describe())


def retornar_resultados(metodo):
    
    lista_df_resultado = []
    for arquivo in ARQUIVOS:

        df_r = pd.read_csv(f"Dados/Resultados/Ranqueado/{metodo}/{arquivo}_métricas.csv")
        lista_df_resultado.append(df_r)

    return lista_df_resultado
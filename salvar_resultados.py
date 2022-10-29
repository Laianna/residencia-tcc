# Constantes
LISTA_CATEGORIAS = ("celulares", "notebooks", "geladeiras", "fogoes", "tvs")


# Salva os resultados gerais (acurácia K e match rank)
def salvar_resultado(nome, df, diretorio):

    df.to_csv(f'{diretorio}/{nome}_métricas.csv', index = False)


# Salva os resultados por categoria (acurácia K e match rank)
def salvar_resultado_categoria(nome, df, diretorio, lista_categorias = LISTA_CATEGORIAS):

    for categoria in lista_categorias:

        df[df["categoria"] == categoria].to_csv(f'{diretorio}/{categoria}/{nome}_métricas.csv', index = False)
import pandas as pd
import numpy as np
import plotly.express as px
# Leitura
def read_diet_and_emission(filename,cols_not_use=[]):
    cols = list(pd.read_csv(filename,nrows=1))
    df = pd.read_csv(filename,usecols=list(col for col in cols if  col not in cols_not_use),
                     dtype={'Entity':'category','Code':'category','Year':'int16'})
    df['Total'] = df.sum(axis=1)
    
    datatypes = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float16) # reduzir o numero de bits dos float
    df = df.astype(datatypes)
    
    df2 = pd.read_csv(filename) # descomentar para saber quao otimizado ficou
    print(f' De {df2.memory_usage().sum()/1000} Kbytes para {df.memory_usage().sum()/1000} Kbytes, uma redução de {(df2.memory_usage().sum()/1000)/(df.memory_usage().sum()/1000)*100:.2f} % ')
    return df


def read_recipes_csv(filename,cols_not_use=[]):
    cols = list(pd.read_csv(filename,nrows=1))
    df = pd.read_csv(filename,usecols=list(col for col in cols if  col not in cols_not_use),
                     dtype={'id':'int32','minutes':'int32','contributor_id':'int32','n_steps':'int8','n_ingredients':'int8'},
                     parse_dates=['submitted'])
    # adicionar os nutrientes divididos por colunas, para facilitar 
    nutricao_list = ['calorias','gordura_total(PVD)','açucar(PVD)','sodio(PVD)','proteina(PVD)','gordura_sat(PVD)','carboidratos(PVD)']
    # O valor diário percentual (PVD) mostra o quanto um nutriente em uma porção de alimento contribui para uma dieta diária total.
    df[nutricao_list] = df.nutrition.str.replace('[','').str.replace(']','').str.split(',',expand=True)
    df[nutricao_list] = df[nutricao_list].astype('float16')
    df.pop('nutrition')
    df2 = pd.read_csv(filename) # descomentar para saber quao otimizado ficou
    print(f' De {df2.memory_usage().sum()/1000} Kbytes para {df.memory_usage().sum()/1000} Kbytes, uma redução de {(df2.memory_usage().sum()/1000)/(df.memory_usage().sum()/1000)*100:.2f} % ')
    return df



def read_reviews_csv(filename,cols_not_use=[]):
    cols = list(pd.read_csv(filename,nrows=1))
    df = pd.read_csv(filename,usecols=list(col for col in cols if  col not in cols_not_use),
                     dtype={'user_id':'int32','recipe_id':'int32','rating':'int8'},
                     parse_dates=['date'])
    df['tam_aval'] = df.review.str.len() # nao nos importa muito o conteudo em sim e so a extenção da avaliacao, uma analise do conteudo é mais complexa e fora de escopo da disciplina
    df['tam_aval'] = df['tam_aval'].fillna(0).astype('int16')
    df.pop('review')
    df2 = pd.read_csv(filename) # descomentar para saber quao otimizado ficou
    print(f' De {df2.memory_usage().sum()/1000} Kbytes para {df.memory_usage().sum()/1000} Kbytes, uma redução de {(df2.memory_usage().sum()/1000)/(df.memory_usage().sum()/1000)*100:.2f} % ')
    return df

# Plots

def barPlot_diet(df,year,countries=['Portugal']):
    df = df[(df['País'].isin(countries) ) & (df['Ano']==year)] # filtrar os dados por ano e paises
    df1 = pd.melt(df, 
        id_vars='País', 
        value_vars=list(df.columns[3:-1]),
        var_name='Tipo', 
        value_name='Consumo(g)'
        ) # converte numa dataframe mais legivel para px, é como um pd.pivot reverso 
    #plot 
    fig = px.bar(df1,x='Consumo(g)',
                width=1200,
                height=500,
                y='País',
                color='Tipo',
                text='Tipo',
                barmode='relative',
                title='Composisão de dieta por pais dividida por tipo de alimento',
                color_discrete_map={
                    'cereais':'#c79c28',
                    'legumes':'#007d2e',
                    "raízes_tubérculos":'#37ed79',
                    "frutos":'#e5ed4e',
                    "lacticínios":'#0786a3',
                    "carne_vermelha":'#962323',
                    "aves":'#fa8989',
                    "ovos":'#d7f595',
                    "marisco":'#95bff5',
                    "leguminosas":'#95f5cb',
                    "frutos_de_casca_rija":'#634135',
                    "óleos":'#778033',
                    "açúcar":'#533380'
                    
    }
                )
   

    fig.update_traces(textposition="inside")
    fig.update_traces(visible=True, )
    fig.write_html('../output/barPlot_diet.html') # guardar em html para manter a interatividade, a conversão depois para png ou jpeg é simples.
    return fig

def barPlot_food_emiss(df):
    df1 = pd.melt(df, 
        id_vars='Alimento', 
        value_vars=list(df.columns[1:10]),
        var_name='Tipo', 
        value_name='Consumo(g)'
        ) 
    px.bar(df1,x='consumo(g)')

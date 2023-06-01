
# modulos de tratar dados
import os
import pandas as pd
import numpy as np

# modulos para visualização

import chart_studio 
import chart_studio.plotly as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express  as px

import plotly.graph_objects as go
from scipy import stats
from plotly.subplots import make_subplots as subplots
# modulos surprise 
from surprise import Dataset,Reader,SVD,KNNWithMeans,KNNBasic,accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from surprise.model_selection import train_test_split


from functions import* 
import warnings

warnings.filterwarnings("ignore")

chart_studio.tools.set_credentials_file(username='Mohaamedl', api_key='MvUpzt7O9SQLxrTEuefC')
sns.set_style('darkgrid')
pd.options.plotting.backend = 'plotly'


# paths dos ficheiros


# filename_diet_animal = '../data/eat-lancet-diet-animal-products.csv' desnecessario visto q esses dados estão ja noutro ficheiro
filename_diet_composition = '../data/eat-lancet-diet-comparison (2).csv'
filename_food_emission = '../data/food-emissions-supply-chain (1).csv'
filename_recipes_raw = 'C:/Users/moham/Documents/data_compressed/RAW_recipes.csv' #! Atenção, deve alterar o caminho para "../data/nome do ficheiro" , usei ficheiros no pc pq nao cabem no github, mas estarei a inclui-los no envio final
filename_reviews_raw= 'C:/Users/moham/Documents/data_compressed/RAW_interactions.csv'
filename_recipes_parq = '../data/recipes.parquet'
filename_reviews_parq = '../data/reviews.parquet'



# funções 

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


def read_recipes_csv(filename,cols_not_use=[],dtypes={'id':'int32','minutes':'int32','n_steps':'int8','n_ingredients':'int8'}):
    cols = list(pd.read_csv(filename,nrows=1))
    df = pd.read_csv(filename,usecols=list(col for col in cols if  col not in cols_not_use),
                     dtype=dtypes,)
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
                     dtype={'user_id':'int32','recipe_id':'int32','rating':'int8'},)
    df['tam_aval'] = df.review.str.len() # nao nos importa muito o conteudo em sim e so a extenção da avaliacao, uma analise do conteudo é mais complexa e fora de escopo da disciplina
    df['tam_aval'] = df['tam_aval'].fillna(0).astype('int16')
    df = df.drop(columns=['review'])
    df2 = pd.read_csv(filename) # descomentar para saber quao otimizado ficou
    print(f' De {df2.memory_usage().sum()/1000} Kbytes para {df.memory_usage().sum()/1000} Kbytes, uma redução de {(df2.memory_usage().sum()/1000)/(df.memory_usage().sum()/1000)*100:.2f} % ')
    return df


# categorias

def classifier(data,lista):
    data = data.replace('[','').replace(']','').replace('\'','').replace(',','').split()
    if not bool(set(data).intersection(lista)):
        tipo = 'vegetariana'
    elif bool(set(data).intersection(lista[:7])):
        if bool(set(data).intersection(lista[:5])) and not bool(set(data).intersection(lista[5:7])) :
            tipo = 'lactovetegariana'
        else:
            tipo = 'ovolactovetariana'
    else:
        tipo = 'não vegetariana'
        
    return tipo



# Plots

def barPlot_diet(df,year,countries=['Portugal']):
    df = df[(df['País'].isin(countries) ) & (df['Ano']==year)] # filtrar os dados por ano e paises
    df1 = pd.melt(df.sort_values('Total'), 
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




def barplot_food_emiss(df,food_list):
        df1 = pd.melt(df.set_index('Alimento').loc[food_list].reset_index().sort_values('Total',ascending=False), 
                id_vars='Alimento', 
                value_vars=list(df.columns[1:10]),
                var_name='tipo_de_perda', 
                value_name='Emissão_CO2(Kg)',
                
                ) 
        plot = px.bar(df1,x='Emissão_CO2(Kg)',color='tipo_de_perda',y='Alimento',
                color_discrete_map={
                'Total':'red',
                'utilização_do_solo':'brown'
                        }
                ,title='Distribuição dos diferentes tipos de emissao por produto')
        plot.show()
        return plot





def test_model(data,modelo,teste_size=0.2):
    print('Testando modelo...')
    t,testset =train_test_split( data,test_size=teste_size)
    predict = modelo.test(testset)
    accuracy.rmse(predict)
    accuracy.mae(predict)
    return print('Feito.')



def get_name(id):
    return df_recipes.loc[df_recipes['id_receita'] == id]['nome'].tolist()[0]



def get_aval(id):
    return df_recipes_2.loc[df_recipes_2['id_receita'] == id]['Avaliação'].tolist()[0]



def get_idx(id):
    return data[data.id_receita == id].index.tolist()[0]





def recommend_sim_coss(id_receita, n):
    print("Recomendando " + str(n) + " receitas similares para: " + get_name(id_receita) + "....")   
    print("-------"*15)  
    recs = results[get_idx(id_receita)][:n]
    ids=[]   
    for i,rec in enumerate(recs): 
        ids.append(rec[1])
        print(f'\n\t({i+1}) : {get_name(rec[1])} ( Similaridade: {rec[0]:.3f}, Avaliação: {get_aval(rec[1]):.1f} )')
    df_rec = df_recipes_2[df_recipes_2.id_receita.isin(ids)]
    return df_rec



def recommend_svd(data,model,id_receita,n):
    ids_receitas = data.id_receita.unique() # pegar as ids todas
    predictions = [model.predict(id_receita, receita2) for receita2 in ids_receitas if receita2 !=id_receita]
    ids_recomendadas_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n] # ordenar as estimações e pegar as n melhores
    ids_recomendadas_n = [prediction.iid for prediction in ids_recomendadas_n] # transformar em ids
    receita = get_name(id_receita)
    receitas = data[data.id_receita.isin(ids_recomendadas_n)].drop_duplicates(subset='nome')
    data_2 = receitas.loc[:,['nome','id_user','id_receita','Avaliação']] # dados para proxima etapa 
    data_2 = data_2.append(df_recipes[df_recipes.id_receita==id_receita]).loc[:,['nome','id_user','id_receita','Avaliação']]
    print(f'As {n} recomendações para a receita {receita} são:\n{receitas.nome.to_list()}')
    return data_2


# -*- coding: utf-8 -*-
"""
Nome: Elisalvo Ribeiro

Análise descritiva de dados do SINASC/CADU

"""

" How do I read tabular data file into pandas
"
" == VER Data Warangling with Pandas Cheat Sheet =="
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('ggplot')
d = pd.read_csv('E:\\Rose\\base_SISVAN_CAD.csv', sep = ",")
d = pd.read_csv('E:\\Rose\\base_SISVAN_CAD.csv', sep = ",", 
                usecols = ['locnasc', 'escolaridade']) # Ler colunas específicas

d = pd.read_csv('E:\\Rose\\base_SISVAN_CAD.csv', sep = ",", 
                usecols = [2, 4]) # Ler colunas específicas

d = pd.read_csv('E:\\Rose\\base_SISVAN_CAD.csv', sep = ",", 
                nrows = 3) # Ler apenas 3 linhas

d = pd.read_csv('E:\\Rose\\base_SISVAN_CAD.csv', sep = ",", 
                dtype = {'idademae':float}) # Ler e já converte o tipo de variavel


" Exploratory Data Analysis - EDA"
type(d)
type(d['pesocat'])
d.pesocat # ou o comando abaixo
d["pesocat"]
d.head() # See first rows
d.shape # Size of DataFrame
d.dtypes # Class of each column in teh DataFrame
d.info() # Are there any missing values in any of the columns
d['pesoQuadrado'] = d.peso^2 # Cria uma nova variavel no banco 

"d.pesocat + ', ' + d.idademae" # concatena 2 variaveis typo character

"Why do some pandas commands end with parenthese, and other commands don't"
d.describe()
d.describe(include = ['object'])

"How do I rename columns in a pandas DataFrame"
d.columns
d.rename(columns = {'pesocat':'Pesocat', 'idademae':'IdadeMae'}, inplace = True)
d.columns = d.columns.str.replace(' ', '_') # Aqui substitui-se espaco entre os rotulos por underline

"How do I remove columns from a pandas DataFrame"
d.drop('locnasc', axis = 1, inplace = True)
d.drop(['IdadeMae', 'estcivmae'], axis = 1, inplace = True)
d.drop([0, 1], axis = 0, inplace = True) # Remove a linha zero e a 1.

"How do I sort a pandas DataFrame or Series"
d.escmae.sort_values()
d['escmae'].sort_values()
d['escmae'].sort_values(ascending = False)
d.sort_values('escmae') # Aqui ordena todo o dataFrame pela variavel 'escmae'
d.sort_values('escmae', ascending = False) 
d.sort_values('escmae', ascending = True) 

"How do I filter rows of a pandas DataFrame by column value"
booleans = []
for length in d.escmae:
    if length >= 2:
        booleans.append(True)
    else:
        booleans.append(False)

booleans[0:50]

is_long = pd.Series(booleans)
is_long.head()
d[is_long]
is_long = d.consultas >= 2
is_long.head()
d[is_long]
d[d.consultas >=2]
d[d.consultas >=2].escmae # Aqui seleciona todos valores da 
                          # variavel consultas >=2 e ordena
                          # pela variavel escmae. os proximos 
                          # comandos faz a mesma coisa
d[d.consultas >= 2]['escmae']
d.loc[d.consultas <= 1, 'escmae']


"How do I apply multiple filter criteria to a pandas DataFrame"
d[(d.consultas >= 2) & (d.class_preterm == 2)]
d[(d.consultas == 2) | (d.class_preterm == 2)]
(d.consultas == 2) | (d.class_preterm == 2) # Cria um vetor booleano (True ou False)

d[(d.class_preterm == 1) | (d.class_preterm == 2) | (d.class_preterm == 3)]
d[d.class_preterm.isin([1, 2, 3])] # Estes 2 comandos fazem a mesma coisa

"""How do DataFrame and Series work with regard to selecting individual entries
and iteration (for X in userdata)?"""
for c in d.locnasc:
    print(c)

for index, row in d.iterrows():
    print(index, row.locnasc, row.pesocat)

"Whats the best way to drop  every non-numeric column from a DataFrame"
d.select_dtypes(include = [np.number]).dtypes

"How I use .describe"
d.describe()
d.describe(include = 'all')
d.describe(include = ['object', 'float64'])

"How do I use the 'axis' parameter in pandas"
d.drop(2, axis = 0).head()
d.mean()
d.mean(axis = 0)
d.mean(axis = 1)
d.mean(axis = 'columns')
d.mean(axis = 1).shape
d.mean(axis = 0).shape
d.consultas.mean()
d.soma_meses_bf.mean()
d.var()

"How do I use string methods in pandas"



"How do I change the data type of a pandas Series?"
d['locnasc'] = d.locnasc.astype(float)


"How should I use a "groupby" in pandas?"
d.groupby('pesocat').mean()
d.groupby('pesocat').locnasc.mean()
d[d.pesocat == 0].locnasc.mean()

d.groupby('pesocat').locnasc.max()
d.groupby('pesocat').locnasc.min()

d.groupby('pesocat').locnasc.agg(['count', 'max', 'min', 'var',
         'std', 'mean', 'median']) # Aqui calcula as estatísticas para a variavel 'locnasc'


%matplotlib inline    
d.groupby('pesocat').mean().plot(kind = 'bar')      
d.groupby('consultas').peso.mean().plot(kind = 'bar')    


"How do I explore a pandas Series?"
d.describe()
d.pesocat.describe()
d.peso.describe()
d.groupby('pesocat').describe()
    
d.pesocat.value_counts()
d.escmae.value_counts()
d.racacorn.value_counts()

d.apgar1.value_counts(normalize = True)
d.apgar1.value_counts(normalize = True).head()
 
d.apgar1.unique()
d.apgar1.nunique()

pd.crosstab(d.pesocat, d.racacorn)
pd.crosstab(d.pesocat, d.consultas)
pd.crosstab(d.consultas, d.escmae)

"ESTUDAR pivot_table"
d.pivot_table(values = 'pesocat', index = 'escmae', columns = 'parto')
d.pivot_table('pesocat', 'parto')


"How do I handle missing value in pandas?"
d.tail()
d.isnull().tail()
d.isnull().head()
d.isnull().sum() # Conta quantos 'NA' tem a variavel
d.isnull().sum(axis = 0)
d.isnull().sum(axis = 1)
pd.Series([True, False, True]).sum()
d[d.escmae.isnull()]

d.dropna(how = 'any').shape # exclue todas as linhas que tem 'NA'
d.dropna(how = 'all').shape # exclue todas as colunas que tem 'NA'
d.dropna(subset = ['pesocat', 'escmae'], how = 'any').shape
d.dropna(subset = ['pesocat', 'escmae'], how = 'all').shape
d['pesocat'].value_counts(dropna = False)
d['pesocat'].fillna(value = 'VARIOUS', inplace = True)

%matplotlib inline
d.peso.plot(kind = 'hist')
d.consultas.value_counts().plot(kind = 'bar')   


"""
What do I need to know about the pandas index?
What are the advantages of using indices instead
of just storing it's values in columns?
"""
d.columns
d.index
d[d.pesocat == 0]
d.loc[45, 'escmae'] # retornar o valor da 45ª linha para a variavel 'escmae'
d.set_index('escmae', inplace = True)
d.head()
d.index
d.columns
d.shape
d.loc[4, 'idademae'] # Pega o indice com valor 4 para a variavel 'idademae'
d.index.name = None
d.head()


d.index.name = 'pesocat'
d.reset_index(inplace = True) # Deu erro!!!!!!!!!!
d.head()

d.describe().index
d.describe().loc['50%', 'peso']
d.idademae.value_counts()[16] # Mostra quantas maes com 16 anos tem na variavel
d.idademae.value_counts().sort_index()

"How do I select multiple rows and columns from a pandas DataFrame?"
d.loc[[10, 45, 80], :] # traz todos os valores das colunas para os indices: 10, 45 e 80
d.loc[:, ['pesocat', 'locnasc']] # traz todos os valores das linhas para as variaveis 'pesocat' e 'locnasc'

d.loc[:, 'pesocat':'locnasc']
d.iloc[:, 0:4] # Traz todos os valores para as colunas de 0 a 4
d.iloc[0:4, ] # Traz todos os valores para as linhas de 0 a 4
d[0:2]
d.ix[0:2, 0:5] # Pega da linha 0 a 2 e da coluna 0 a 5


"Whem should I use the "inplace' parameter in pandas?"


"How do I make my pandas DataFrame smaller and faster?"
d.info(memory_usage = 'deep')
d.memory_usage()
d.memory_usage(deep = True)
d.memory_usage(deep = True).sum()
sorted(d.escmae.unique())
d['escmae'] = d.escmae.astype('category')
d.dtypes

#How do I use pandas with scikit-learn to create Kaggle submissions?
features_cols = ['locnasc', 'idademae', 'estcivmae', 'escmae', 'consultas', 'semagestac',
                 'total_meses_bf', 'nasc_res']
features2_cols = ['pesocat','locnasc', 'idademae', 'estcivmae', 'escmae', 'consultas', 'semagestac',
                 'total_meses_bf', 'nasc_res']
## === OBS: COMO TRANSFORMAR TODAS ESTAS VARIAVEIS EM CATEGORICAS ?????????
features_cols['locnasc', 'estcivmae', 'escmae', 'nasc_res'] = features_cols.astype([:,'category'])
df = d.loc[:, features2_cols]
x = d.loc[:, features_cols]
x.shape
y = d.pesocat
y.shape

train = df.sample(frac = .7, random_state = 99)
test = df.loc[~x.index.isin(train.index), :]

x_train = train.loc[:, features_cols]
y_train = train.pesocat
x_test = test.loc[:, features_cols]
y_test = test.pesocat

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

new_predict_class = logreg.predict(x_test) 

# How do I create dummy variables in pandas?
d['sexo'] = d.sexo.map({0:'female', 1:'male'})

df1 = pd.get_dummies(d.escmae)
df1 = pd.get_dummies(d.escmae, prefix = 'escmae')
df3 = pd.get_dummies(d.escmae).iloc[:,1:] # Exclui a primeira coluna de dummies
df3 = pd.get_dummies(d.escmae, prefix = 'sexo').iloc[:,1:] # Adiciona o prefixo antes do valor
df3 = pd.get_dummies(d.escmae).iloc[:,2:] # Exclui até a 2ª coluna de dummies

" ADICIONANDO COLUNAS TRATADAS"
d = pd.concat([d, df1], axis = 1)
df4 = pd.get_dummies(d, columns = ['locnasc', 'estcivmae']) 

#Dropo a 1ª dummie criada
df4 = pd.get_dummies(d, columns = ['locnasc', 'estcivmae'], drop_first = True) 

" CRIANDO VARIAVEL "
d['peso_dobro'] = (d.peso)*2

"How do I work with dates and times in pandas?"


"How do I find and remove duplicates rows in pandas?"
d.idademae.duplicated() # Retornar um vetor de 'True' ou 'False'
d.idadepai.duplicated().sum()
d.loc[d.duplicated(keep = 'first'), :]
d.loc[d.duplicated(keep = 'last'), :]
d.loc[d.duplicated(keep = False), :]
d.drop_duplicates(keep = 'first').shape
d.drop_duplicates(keep = False).shape
d.duplicated(subset = ['idademae', 'idadepai']).sum()
d.drop_duplicates(subset = ['idademae','idadepai']).shape


"""How do I change display options in pandas?
Is there a simple way to format large number with commas (1,000,000)
when printing or graphing (on the axes)?"""

pd.set_option('display.max_rows')
pd.reset_option('display.max_rows')
pd.set_option('display.max_rows', None)
pd.get_option('display.max_columns')
pd.get_option('display.max_colwidth')





d.plot(x = "idadepai", y = "idademae", kind = "scatter",
       figsize = [15, 10], color = "b", alpha = 0.3,
       fontsize = 14)
plt.title("idade do pai x idade da mãe", fontsize = 24,
        color = "darkred")
plt.xlabel("idade do pai", fontsize = 18)
plt.ylabel("idade da mãe", fontsize = 18)
plt.show()

corr = d.corr()
plt.figure(figsize = (9, 7))
sns.heatmap(corr, cmap = "RdBu",
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values)
plt.show()


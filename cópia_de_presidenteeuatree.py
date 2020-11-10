
#Referenciando as dependências
import pandas as pd 
import numpy as np 
from sklearn import tree 
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import graphviz 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()  # estilo do gráfico

# Carregamos o banco de dados 
eua = pd.read_csv('https://raw.githubusercontent.com/rfdiego/Inteligencia-Artificial-Votacao/main/president_county_candidate.csv', sep=',')
datasetoriginal = eua #copiamos o dataset para usarmos de referencia nas features e classes
eua

# informações das variaveis 
eua.info()

#transformando a variavel target em numerica 
labelencoder_X = LabelEncoder()
eua['NCandidate'] = labelencoder_X.fit_transform(eua['candidate'])
eua['NCandidate'].value_counts()

atributos_continuos = ['votes']
atributos_categoricos = ['state','county','party']
for col in atributos_categoricos:
    dummies = pd.get_dummies(eua[col], prefix=col)
    eua = pd.concat([eua, dummies], axis=1)
    eua.drop(col, axis=1, inplace=True)
eua.head()

# Eliminando as colunas repetidas
eua.drop(['candidate'],axis=1,inplace=True)

# informações das novas variaveis 
myfeatures = eua
eua.head()


# separando o treino e o teste
X_train, X_test, y_train, y_test = train_test_split( eua.drop('NCandidate',axis=1), #pega todos valores menos candidatos
                                                     eua['NCandidate'], test_size=0.25) #pega somente os candidatos
clf = tree.DecisionTreeClassifier(criterion='entropy') # usar o maximo nivel para melhor resutado, retirando parametro , max_depth=8
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("\nMatriz de confusão detalhada:\n",
      pd.crosstab(y_test, predictions, rownames=['Real'], colnames=['Predito'],
      margins=True, margins_name='Todos'))
	  
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

'''
#decidir qual melhor valor do nivel da arvore
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    scores = cross_val_score(t, X_test, y_test, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std()*2))
'''

print(list(myfeatures.drop('NCandidate',axis=1)))
	

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("eua")
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names= list(myfeatures.drop('NCandidate',axis=1)),
                                
                         class_names= datasetoriginal['candidate'],  
                         filled=True, rounded=True, 
                         special_characters=True)
graph = graphviz.Source(dot_data, format="png") 
graph

#salvar o Gráfico da arvore para ilustração
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = list(myfeatures.drop('NCandidate',axis=1)),
										class_names=datasetoriginal['candidate'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('my_decision_tree2.png')

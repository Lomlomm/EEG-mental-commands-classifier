from pathlib import Path
import numpy as np
import pandas as pd

# Get the path of EEG csv file
p = Path('data/Records/Raw/Processed/EEG.csv')

# carga los datos
#Importamos el dataset para iniciar el análisis
eeg = pd.read_csv(p)

#Visualizamos los primeros 5 datos del dataset
print(eeg.head())

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

print('Descripción del dataset:')
print(eeg.describe())

print('Distribución de Intencion de movimiento mano derecha  0=abrir,  1=cerrar:')
print(eeg.groupby('Classification').size())

#  PARA GRAFICAR ADELANTE LAS MATRICES DE CONFUSION
import matplotlib.pyplot as plt

#importar metrics para usar la matriz de confusión
from sklearn import metrics

# Modelo de máquinas de Soporte
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Separamos todos los datos con las características y las etiquetas o resultados
#  eje 0= filas, eje 1=columnas  (segundo parámetro de función drop)
X = np.array(eeg.drop(['Classification'], axis=1))
y = np.array(eeg['Classification'])

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

#Modelo de Máquinas de Vectores de Soporte
algoritmo = SVC(C=0.85, kernel='linear')
algoritmo.fit(X_train, y_train)
Y_pred_SVM = algoritmo.predict(X_test)
# SVM Precisión entrenamiento
print(' SVM  Entrenamiento')
print('Precisión Máquinas de Vectores de Soporte: {}'.format(algoritmo.score(X_train, y_train)))
print(Y_pred_SVM)

print('Distribución de Intencion de movimiento mano derecha  0=abrir,  1=cerrar:')
todos = np.count_nonzero(Y_pred_SVM)
unos=np.sum(Y_pred_SVM)
unos_arr = np.bincount(Y_pred_SVM)

import collections
print('Cerrar : ',unos)
print('[0=Abrir  1=Cerrar]')
print(unos_arr)
print(' ceros en prediccion: ',collections.Counter(Y_pred_SVM)[0])
print(' unos en predicción:  ',collections.Counter(Y_pred_SVM)[1])

# MATRIZ DE CONFUSIÓN.

# creamos la matriz de confusion
confusion_matrix = metrics.confusion_matrix(y_test,Y_pred_SVM)


# presentar claramente la matriz de confusion  
# para eso la convertimos a tabla
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])


# como queremos imprimir la matriz de confusion, 
# importamos librería para graficar
import matplotlib.pyplot as plt


# Desplegamos la matriz de confusión
cm_display.plot()
plt.show()
input("Pulsa una tecla para continuar...")
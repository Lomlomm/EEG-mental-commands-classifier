import os
import numpy as np
import pandas as pd
from datetime import datetime
import collections
# import warnings filter
from warnings import simplefilter
#  PARA GRAFICAR ADELANTE LAS MATRICES DE CONFUSION
import matplotlib.pyplot as plt

#importar metrics para usar la matriz de confusión
from sklearn import metrics

# Modelo de máquinas de Soporte
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# como queremos imprimir la matriz de confusion, 
# importamos librería para graficar
import matplotlib.pyplot as plt

def get_parameters():
    kernels = ['rbf', 'linear', 'poly']
    # Obtener los parámetros de entrada aleatoriamente
    # C es un numero entre 1.0 y 3.0
    C = np.random.uniform(1.0, 3.1)
    return C, kernels[np.random.randint(0, 3)]

def getSVM(C, kernel, meta):
    # Obtener la fecha y hora actual en formato YYYYMMDD_HHMMSS
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Modelo de Máquinas de Vectores de Soporte
    algoritmo = SVC(C=C, kernel=kernel)
    algoritmo.fit(X_train, y_train)
    Y_pred_SVM = algoritmo.predict(X_test)
    # SVM Precisión entrenamiento
    score = algoritmo.score(X_train, y_train)
    if score < meta:
        return False
    print('Precisión Máquinas de Vectores de Soporte: {}'.format(score))

    print('Distribución de Falla cardíaca Estimada  0=no,  1=Sí:')
    unos=np.sum(Y_pred_SVM)
    unos_arr = np.bincount(Y_pred_SVM)

    print('FALLA : ',unos)
    print('[0=No Falla  1=FALLA]')
    print(unos_arr)
    print(' ceros en prediccion: ',collections.Counter(Y_pred_SVM)[0])
    print(' unos en predicción:  ',collections.Counter(Y_pred_SVM)[1])

    # MATRIZ DE CONFUSIÓN.
    # creamos la matriz de confusion
    confusion_matrix = metrics.confusion_matrix(y_test,Y_pred_SVM)

    # presentar claramente la matriz de confusion  
    # para eso la convertimos a tabla
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    # Verificamos si existe la carpeta para guardar las imagenes
    # si no existe la creamos
    if not os.path.exists('img'):
        os.makedirs('img')

    # Verificamos si existe la carpeta rbf para guardar las imagenes dentro de img
    # si no existe la creamos
    if not os.path.exists('img/rbf'):
        os.makedirs('img/rbf')

    # Verificamos si existe la carpeta linear para guardar las imagenes dentro de img
    # si no existe la creamos
    if not os.path.exists('img/linear'):
        os.makedirs('img/linear')

    # Verificamos si existe la carpeta poly para guardar las imagenes dentro de img
    # si no existe la creamos
    if not os.path.exists('img/poly'):
        os.makedirs('img/poly')

    # Desplegamos la matriz de confusión
    cm_display.plot()
    plt.title(f"Matriz de Confusión\nC={C}, kernel='{kernel}'\nScore={score:.4f}")
    # Guardamos la imagen en la carpeta img
    plt.savefig(f'img/{kernel}/{now}.png')

    return score

# carga los datos
#Importamos el dataset para iniciar el análisis
heart = pd.read_csv("heart.csv")

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#Separamos todos los datos con las características y las etiquetas o resultados
#  eje 0= filas, eje 1=columnas  (segundo parámetro de función drop)
X = np.array(heart.drop(['HeartDisease'], axis=1))
y = np.array(heart['HeartDisease'])

precision_esperada = float(input("Ingrese la precisión esperada: "))
iteraciones = int(input("Ingrese el número de iteraciones: "))

for i in range(iteraciones + 1):
    while True:
        C, kernel = get_parameters()
        score = getSVM(C, kernel, precision_esperada)
        if score:
            print(f"C: {C}, kernel: {kernel}, score: {score}")
            break

from keras.models import Sequential
from keras.layers import Dense
from pathlib import Path
import numpy

#tf.keras.backend.clear_session()  # Reseteo sencillo
#from tensorflow.keras import utils
#from tensorflow.keras import layers

model = Sequential()
model.add(Dense(20, input_dim=11, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

import warnings
warnings.filterwarnings('ignore')

# Fija las semillas aleatorias para la reproductibilidad
numpy.random.seed(7)

# carga los datos
pe = Path('Examples/heartE.csv')
dataset1 = numpy.loadtxt(pe, delimiter=",")

# DATOS PARA ENTRENAMIENTO 
# dividido en variables de entrada (X) y salida (Y)
X = dataset1[:,0:11]
Y = dataset1[:,11]

print(X)
print(Y)

#  DATOS PARA VALIDACIÓN
pv = Path('Examples/heartV.csv')
dataset2 = numpy.loadtxt(pv, delimiter=",")

# dividido en variables de entrada (W) y salida (Z)
W = dataset2[:,0:11]
Z = dataset2[:,11]
print(W)
print(Z)

# Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajusta el modelo
# SE LE CARGAN LOS DATOS DE ENTRENAMIENTO
model.fit(X, Y, epochs=500, batch_size=10)

# VALIDACIÓN
# test del modelo. con los nuevos datos
test_loss, test_acc = model.evaluate(W, Z)
print('Precision del test:', test_acc)

# calcula las predicciones
predictions = model.predict(W)
print(W)
print(predictions)

# redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
print(rounded)

#importar metrics para usar la matriz de confusión
from sklearn import metrics

# creamos la matriz de confusion
confusion_matrix = metrics.confusion_matrix(Z,rounded)

# presentar claramente la matriz de confusion  
# para eso la convertimos a tabla
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

# como queremos imprimir la matriz de confusion, 
# importamos librería para graficar
import matplotlib.pyplot as plt

# Desplegamos la matriz de confusión
cm_display.plot()
plt.show()

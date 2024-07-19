import convert_json_to_pd

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from flask import jsonify
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def load_data(data):
    # Separar características y etiquetas
    X = data.drop(columns=['Classification', 'Time:512Hz']).values
    y = data['Classification'].values

    # Codificar las etiquetas a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    mapping = {original_label: encoded_value for original_label, encoded_value in zip(y, y_encoded)}
    return X, y_encoded


def compute_covariance_matrices(X, y):
    # Calcular las matrices de covarianza para cada clase
    classes = np.unique(y)
    print(classes)
    cov_matrices = []

    for class_label in classes:
        class_indices = np.where(y == class_label)[0]
        class_data = X[class_indices]
        cov_matrices.append(np.cov(class_data.T))
    return cov_matrices


def compute_csp_projection(cov_matrices):
    # Calcular las proyecciones de CSP
    num_csp_components = 8 # Seleccionar el número de componentes CSP

    # Calcular la matriz de covarianza promedio
    avg_cov_matrix = sum(cov_matrices) / len(cov_matrices)
    # Calcular la matriz generalizada de eigenvalores y eigenvectores
    eigvals, eigvecs = eigh(avg_cov_matrix)
    

    # Seleccionar los eigenvectores correspondientes a los Mínimos y Máximos autovalores
    idx_min = np.argsort(eigvals)[:num_csp_components]
    idx_max = np.argsort(eigvals)[-num_csp_components:]

    # Concatenar los eigenvectores
    w = np.hstack([eigvecs[:, idx_min], eigvecs[:, idx_max]])

    print('Projection matrix length ', len(w))
    return w


def apply_csp_projection(X, w):
    # Aplicar la proyección CSP a los datos
    X_csp = np.dot(X, w)
    print('Initial shape of matrix', X.shape)
    print('matrix projected shape (rows, columns)', X_csp.shape)


    return X_csp

def main():
    # Inicializar un objeto de escalado
    scaler = StandardScaler()
    # Cargar los datos
    df_training_data = convert_json_to_pd.Convert2DF('https://api-flask-tesina-2745b2978945.herokuapp.com/processData')
    df_evaluation_data = convert_json_to_pd.Convert2DF('https://api-flask-tesina-2745b2978945.herokuapp.com/getEvaluationData')
    print(df_training_data['Classification'].value_counts())

    X, y = load_data(df_training_data)

    # Calcular las matrices de covarianza para cada clase
    cov_matrices = compute_covariance_matrices(X, y)
    # Calcular la proyección CSP
    w = compute_csp_projection(cov_matrices)
    # Aplicar la proyección CSP a los datos
    X_csp = apply_csp_projection(X, w)

    X_evaluation, y_evaluation = load_data(df_evaluation_data)

    cov_matrices_evaluation = compute_covariance_matrices(X_evaluation, y_evaluation)
    cov_matrices_evaluation.pop(1)
    w_evaluation = compute_csp_projection(cov_matrices_evaluation)
    X_csp_evaluation = apply_csp_projection(X_evaluation, w_evaluation)

    X_evaluation_scaled_fit = scaler.fit_transform(X_csp_evaluation)
    X_evaluation_scaled = scaler.transform(X_evaluation_scaled_fit)

    # rus = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = rus.fit_resample(X_csp, y)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_csp, y, test_size=0.2, random_state=42)
    
    # Ajustar el escalador a tus datos de entrenamiento y transformar los datos
    X_train_scaled = scaler.fit_transform(X_train)

    # Aplicar el mismo escalado a los datos de prueba
    X_test_scaled = scaler.transform(X_test)

    # Entrenar una Máquina de Soporte Vectorial (SVM) 
    svm_classifier = SVC(kernel='rbf', C=2, gamma=0.15, probability=True, class_weight='balanced')
    svm_classifier.fit(X_train_scaled, y_train)

    # Realizar predicciones
    y_pred = svm_classifier.predict(X_test_scaled)
    y_pred_evaluation = svm_classifier.predict(X_evaluation_scaled)

    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # cv_scores = cross_val_score(svm_classifier, X_csp, y, scoring='accuracy')
    # print("Scores de Validación Cruzada:", cv_scores)
    # print("Promedio de Scores de Validación Cruzada:", cv_scores.mean())

    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred))
    # scores = StratifiedKFold(svm_classifier, X_train_scaled, y_train) # Aquí puedes especificar el número de folds para la validación cruzada

    # print("Scores de Validación Cruzada:", scores)
    # print("Promedio de Scores de Validación Cruzada:", scores.mean())

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel('Classifier Assigned Classes')
    plt.ylabel('Real Classes')
    plt.title('Confusion Matrix')
    plt.show()

    # Obtener las puntuaciones de las predicciones del modelo
    y_scores = svm_classifier.decision_function(X_test_scaled)
    # Calcula la precisión y el recall para diferentes umbrales de decisión
    # Obtener las probabilidades para cada clase en lugar de las predicciones directas de clase
    y_probs = svm_classifier.predict_proba(X_test_scaled)

    # Calcular la precisión y el recall para cada clase
    precision = dict()
    recall = dict()
    n_classes = 3
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_probs[:, i])

    # # Graficar una Curva de Precisión-Recall para cada clase
    # plt.figure(figsize=(8, 6))
    # plt.plot(recall[0], precision[0], marker='.', label=f'Izquierda')
    # plt.plot(recall[1], precision[1], marker='.', label=f'Descanso')
    # plt.plot(recall[2], precision[2], marker='.', label=f'Derecha')


    # plt.xlabel('Recobro')
    # plt.ylabel('Precisión')
    # plt.title('Curva de Precisión-Recobro para clases \'izquierda\', \'descanso\', \'derecha\' ')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # title = "Learning Curves (SVM, RBF kernel)"
    # plt = plot_learning_curve(svm_classifier, title, X, y, cv=10, n_jobs=-1)
    # plt.show()

    arr_list = y_pred.tolist()
    arr_list.sort(reverse = True)
    return jsonify(arr_list)
    
if __name__ == '__main__':
    main()
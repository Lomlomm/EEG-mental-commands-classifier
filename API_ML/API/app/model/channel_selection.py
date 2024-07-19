
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.linalg import eigh
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(file_path)
    # Separar características y etiquetas
    X = data.drop(columns=['Classification', 'Time:512Hz']).values
    y = data['Classification'].values

    # Codificar las etiquetas a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    mapping = {original_label: encoded_value for original_label, encoded_value in zip(y, y_encoded)}
    print(mapping)
    return X, y_encoded


def compute_covariance_matrices(X, y):
    # Calcular las matrices de covarianza para cada clase
    classes = np.unique(y)
    cov_matrices = []

    for class_label in classes:
        class_indices = np.where(y == class_label)[0]
        class_data = X[class_indices]
        cov_matrices.append(np.cov(class_data.T))

    return cov_matrices


def compute_csp_projection(cov_matrices):
    # Calcular las proyecciones de CSP
    num_channels = cov_matrices[0].shape[0]
    num_csp_components = 10  # Seleccionar el número de componentes CSP

    # Calcular la matriz de covarianza promedio
    avg_cov_matrix = sum(cov_matrices) / len(cov_matrices)

    # Calcular la matriz generalizada de eigenvalores y eigenvectores
    eigvals, eigvecs = eigh(avg_cov_matrix)

    # Seleccionar los eigenvectores correspondientes a los Mínimos y Máximos autovalores
    idx_min = np.argsort(eigvals)[:num_csp_components]
    idx_max = np.argsort(eigvals)[-num_csp_components:]

    # Concatenar los eigenvectores
    w = np.hstack([eigvecs[:, idx_min], eigvecs[:, idx_max]])

    return w


def apply_csp_projection(X, w):
    # Aplicar la proyección CSP a los datos
    X_csp = np.dot(X, w)
    return X_csp

if __name__ == '__main__':
        
    # Cargar los datos
    file_path = './API_ML/API/app/model/data/cube_data/Processed/concatenated_data.csv'
    X, y = load_data(file_path)

    # Calcular las matrices de covarianza para cada clase
    cov_matrices = compute_covariance_matrices(X, y)
    # Calcular la proyección CSP
    w = compute_csp_projection(cov_matrices)
    # Aplicar la proyección CSP a los datos
    X_csp = apply_csp_projection(X, w)


    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_csp, y, test_size=0.2, random_state=42)
    # Inicializar un objeto de escalado
    scaler = StandardScaler()
    # Ajustar el escalador a tus datos de entrenamiento y transdformar los datos
    X_train_scaled = scaler.fit_transform(X_train)

    # Aplicar el mismo escalado a los datos de prueba
    X_test_scaled = scaler.transform(X_test)

    # Entrenar una Máquina de Soporte Vectorial (SVM)
    svm_classifier = SVC(kernel='rbf', C=2, gamma=1.5)
    with tqdm(total=100, desc="Ajuste del clasificador SVM") as pbar:
        svm_classifier.fit(X_train_scaled, y_train)
        pbar.update(100)
    # Realizar predicciones
    y_pred = svm_classifier.predict(X_test_scaled)

    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred))

    
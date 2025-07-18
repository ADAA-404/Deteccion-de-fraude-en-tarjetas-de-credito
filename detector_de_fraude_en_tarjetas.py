"""
detector_de_fraude_en_tarjetas.py

Este script implementa un sistema de detección de fraude en transacciones de tarjetas de crédito.
Compara el rendimiento y la velocidad de entrenamiento de algoritmos de Machine Learning
(Árboles de Decisión y Máquinas de Vectores de Soporte) implementados en Scikit-learn
y Snap ML, una librería optimizada para alto rendimiento.

El flujo de trabajo incluye:
- Carga y preparación de un dataset de transacciones.
- Inflado del dataset para simular un volumen mayor de datos.
- Análisis de la distribución de clases y montos de transacciones.
- Preprocesamiento de datos (escalado y normalización).
- División de datos en conjuntos de entrenamiento y prueba.
- Entrenamiento y evaluación comparativa de modelos de Scikit-learn y Snap ML.
- Cálculo de métricas de evaluación clave como ROC-AUC y Hinge Loss.

Librerías principales utilizadas: pandas, numpy, scikit-learn, matplotlib, seaborn, snapml.
"""

import warnings
warnings.filterwarnings('ignore') # Suprime advertencias para una salida más limpia

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os # Importa el módulo 'os' para interactuar con el sistema operativo

import requests # Para realizar solicitudes HTTP (descargar archivos)
import tarfile  # Para trabajar con archivos .tgz

# Nota: %matplotlib inline es un "magic command" de IPython/Jupyter y no funciona en scripts .py puros.
# Si lo ejecutas como un script, matplotlib.pyplot.show() es suficiente para mostrar los gráficos.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, hinge_loss
import time
import gc # Garbage Collector para liberar memoria
import sys # Para obtener información del sistema
from typing import Tuple

# Importar las implementaciones de modelos desde Scikit-learn
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.svm import LinearSVC as SklearnLinearSVC

# Importar las implementaciones de modelos desde Snap ML
# Asegúrate de haber instalado snapml: pip install snapml
try:
    from snapml import DecisionTreeClassifier as SnapMLDecisionTreeClassifier
    from snapml import SupportVectorMachine as SnapMLSupportVectorMachine
    SNAPML_AVAILABLE = True
except ImportError:
    print("Advertencia: Snap ML no está instalado. Solo se ejecutarán los modelos de Scikit-learn.")
    SNAPML_AVAILABLE = False


# --- Funciones para el flujo de detección de fraude ---

def download_and_load_data(url: str, filename: str) -> pd.DataFrame:
    """
    Descarga un archivo .tgz desde una URL y lo carga como un DataFrame CSV.
    Si el archivo ya existe, lo carga directamente.

    Args:
        url (str): La URL del archivo .tgz a descargar.
        filename (str): El nombre del archivo CSV esperado dentro del .tgz.

    Returns:
        pd.DataFrame: El DataFrame de pandas con los datos cargados.
                      Retorna un DataFrame vacío si la descarga o carga falla.
    """
    print(f"Intentando descargar y cargar el dataset desde: {url}")
    
    # Verificar si el archivo CSV ya existe localmente
    if not os.path.exists(filename):
        print(f"El archivo '{filename}' no se encontró localmente. Intentando descargarlo y extraerlo...")
        try:
            # Descargar el archivo TGZ
            tgz_filename = url.split('/')[-1] # Obtener el nombre del archivo tgz de la URL
            print(f"Descargando {tgz_filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status() # Lanza una excepción para errores HTTP (4xx o 5xx)

            # Guardar el archivo TGZ
            with open(tgz_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"'{tgz_filename}' descargado exitosamente.")

            # Descomprimir el archivo TGZ
            print(f"Extrayendo '{filename}' de '{tgz_filename}'...")
            with tarfile.open(tgz_filename, "r:gz") as tar:
                tar.extractall() # Extrae todos los archivos del .tgz al directorio actual
            print(f"'{filename}' extraído correctamente.")

            # Opcional: Eliminar el archivo .tgz después de la extracción
            os.remove(tgz_filename)
            print(f"Archivo temporal '{tgz_filename}' eliminado.")

        except requests.exceptions.RequestException as e:
            print(f"Error de red o HTTP al descargar el dataset: {e}")
            return pd.DataFrame() # Retorna DataFrame vacío si falla la descarga
        except tarfile.ReadError as e:
            print(f"Error al leer o descomprimir el archivo .tgz: {e}")
            return pd.DataFrame() # Retorna DataFrame vacío si falla la descompresión
        except Exception as e:
            print(f"Un error inesperado ocurrió durante la descarga/extracción: {e}")
            return pd.DataFrame() # Retorna DataFrame vacío para otros errores

    # Intentar cargar el archivo CSV después de asegurar que existe
    try:
        raw_data = pd.read_csv(filename)
        print(f"Se encontraron {len(raw_data)} observaciones en el dataset de fraude de tarjetas de crédito.")
        print(f"Se encontraron {len(raw_data.columns)} variables en el dataset.")
        print("Primeras filas del dataset:")
        print(raw_data.head())
        return raw_data
    except FileNotFoundError:
        # Esto no debería ocurrir si la lógica de descarga/extracción funcionó
        print(f"Error fatal: El archivo '{filename}' no se encontró incluso después de la descarga/extracción.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al leer el archivo '{filename}': {e}")
        return pd.DataFrame()

def inflate_dataset(df: pd.DataFrame, n_replicas: int = 10) -> pd.DataFrame:
    """
    Infla el dataset original replicando sus filas 'n_replicas' veces.
    Esto se usa para simular un dataset más grande para pruebas de rendimiento.

    Args:
        df (pd.DataFrame): El DataFrame original.
        n_replicas (int): Número de veces que se replicará el dataset.

    Returns:
        pd.DataFrame: El DataFrame inflado.
    """
    print(f"\n--- Inflado del Dataset (x{n_replicas}) ---")
    big_raw_data = pd.DataFrame(np.repeat(df.values, n_replicas, axis=0), columns=df.columns)
    print(f"Se encontraron {len(big_raw_data)} observaciones en el dataset inflado.")
    print(f"Se encontraron {len(big_raw_data.columns)} variables en el dataset.")
    print("Primeras filas del dataset inflado:")
    print(big_raw_data.head())
    return big_raw_data

def analyze_class_and_amount_distribution(df: pd.DataFrame):
    """
    Realiza un análisis exploratorio de la distribución de la variable objetivo (Clase)
    y la distribución de los montos de las transacciones.

    Args:
        df (pd.DataFrame): El DataFrame a analizar.
    """
    print("\n--- Análisis de Distribución de Clase y Monto ---")
    # Gráfico de pastel de la distribución de clases
    labels = df.Class.unique()
    sizes = df.Class.value_counts().values
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors=['skyblue', 'lightcoral'])
    ax.set_title('Distribución de la Variable Objetivo (Clase)')
    ax.axis('equal') # Asegura que el pastel sea un círculo
    plt.show()

    # Histograma de montos de transacciones
    plt.figure(figsize=(8, 5))
    plt.hist(df.Amount.values, bins=50, histtype='bar', facecolor='g', alpha=0.7)
    plt.title('Distribución de Montos de Transacción')
    plt.xlabel('Monto de Transacción')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    print(f"Valor mínimo del monto de transacción: {np.min(df.Amount.values):.2f}")
    print(f"Valor máximo del monto de transacción: {np.max(df.Amount.values):.2f}")
    print(f"El 90% de las transacciones tienen un monto menor o igual a: {np.percentile(df.Amount.values, 90):.2f}")

def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa las características del DataFrame:
    - Escala las columnas V1-V28 usando StandardScaler.
    - Normaliza la matriz de características X usando la norma L1.
    - Separa las características (X) de las etiquetas (y).

    Args:
        df (pd.DataFrame): El DataFrame con los datos brutos o inflados.

    Returns:
        tuple[np.ndarray, np.ndarray]: Un tuple que contiene la matriz de características (X)
                                       y el vector de etiquetas (y), ambos como arrays de NumPy.
    """
    print("\n--- Preprocesamiento de Características (Escalado y Normalización) ---")
    # Escalar las columnas de características V1 a V28
    # El Time y el Amount usualmente se tratan diferente o se excluyen.
    # En este caso, el Time se excluye en la definición de X, y Amount se escala junto a V's.
    # El código original escala desde la columna 1 hasta la 30, lo que incluye 'Amount'.
    # Si 'Time' es la columna 0, y 'Amount' es la 29, entonces V1-V28 y Amount.
    # Se sigue la lógica original: columnas 1 a 30 (excluyendo la columna 'Class' que es la 30 para Python).
    
    # Identificar la columna 'Class' y la columna 'Time'
    # Asumiendo que 'Time' es la primera columna y 'Class' es la última.
    
    # Copia para evitar SettingWithCopyWarning
    processed_data = df.copy() 
    
    # Las columnas V1-V28 y Amount deben ser escaladas
    # Las columnas 'V#' van de df.columns[1] a df.columns[29] (inclusive 'Amount' si sigue el patrón).
    # En el dataset, 'Time' es la columna 0, 'V1'...'V28' son 1 a 28, 'Amount' es 29, 'Class' es 30.
    
    # Identificar las columnas a escalar (V1 a V28, Amount)
    # Excluir 'Time' (columna 0) y 'Class' (columna 30)
    cols_to_scale = [col for col in processed_data.columns if col.startswith('V') or col == 'Amount']
    
    if 'Time' in processed_data.columns:
        # El código original parece escalar hasta la columna 29 (índice 28)
        # y luego usar X = data_matrix[:, 1:30], lo que efectivamente toma V1-V28 y Amount.
        # Vamos a seguir la lógica original de `iloc[:, 1:30]` que cubre V1-V28 y Amount.
        processed_data.iloc[:, 1:30] = StandardScaler().fit_transform(processed_data.iloc[:, 1:30])
    else:
        # Si 'Time' no está, escalamos todas excepto 'Class'
        cols_for_scaling_all = [col for col in processed_data.columns if col != 'Class']
        processed_data[cols_for_scaling_all] = StandardScaler().fit_transform(processed_data[cols_for_scaling_all])


    data_matrix = processed_data.values

    # X: matriz de características (excluye la variable 'Time' - columna 0)
    # y: vector de etiquetas (columna 'Class' - última columna)
    X = data_matrix[:, 1:30] # Columnas V1 a V28 y Amount
    y = data_matrix[:, 30]  # Columna 'Class' (índice 30)

    # Normalización de los datos (norma L1)
    X = normalize(X, norm="l1")

    print(f'Forma de X (características): {X.shape}')
    print(f'Forma de y (etiquetas): {y.shape}')

    # Liberar memoria de los DataFrames grandes si ya no son necesarios
    del df, processed_data # Eliminar el df original y el temporal
    gc.collect() # Forzar la recolección de basura

    return X, y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> tuple:
    """
    Divide las características y etiquetas en conjuntos de entrenamiento y prueba.
    Utiliza estratificación para mantener la proporción de clases.

    Args:
        X (np.ndarray): Matriz de características.
        y (np.ndarray): Vector de etiquetas.
        test_size (float): Proporción del dataset a incluir en la división de prueba.
        random_state (int): Semilla para la reproducibilidad de la división.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) como arrays de NumPy.
    """
    print("\n--- División de Datos en Entrenamiento y Prueba ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f'Forma de X_train: {X_train.shape}, Forma de Y_train: {y_train.shape}')
    print(f'Forma de X_test: {X_test.shape}, Forma de Y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test

def compare_decision_trees(X_train, y_train, X_test, y_test):
    """
    Compara el rendimiento de Decision Tree Classifiers de Scikit-learn y Snap ML.

    Args:
        X_train, y_train, X_test, y_test: Conjuntos de datos de entrenamiento y prueba.
    """
    print("\n--- Comparación de Árboles de Decisión (Scikit-learn vs. Snap ML) ---")
    # Calcular pesos de muestra para manejar el desbalance de clases
    w_train = compute_sample_weight('balanced', y_train)

    # --- Scikit-learn Decision Tree ---
    sklearn_dt = SklearnDecisionTreeClassifier(max_depth=4, random_state=35)
    t0 = time.time()
    sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
    sklearn_time = time.time() - t0
    print(f"[Scikit-Learn Decision Tree] Tiempo de entrenamiento (s): {sklearn_time:.5f}")

    # --- Snap ML Decision Tree ---
    if SNAPML_AVAILABLE:
        snapml_dt = SnapMLDecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4) # n_jobs=-1 para todos los núcleos
        t0 = time.time()
        snapml_dt.fit(X_train, y_train, sample_weight=w_train)
        snapml_time = time.time() - t0
        print(f"[Snap ML Decision Tree] Tiempo de entrenamiento (s): {snapml_time:.5f}")

        # Comparación de velocidad
        training_speedup = sklearn_time / snapml_time
        print(f'[Decision Tree] Aceleración Snap ML vs. Scikit-Learn: {training_speedup:.2f}x')

    # Evaluación de ROC-AUC
    sklearn_pred_proba = sklearn_dt.predict_proba(X_test)[:, 1]
    sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred_proba)
    print(f'[Scikit-Learn Decision Tree] Puntuación ROC-AUC: {sklearn_roc_auc:.3f}')

    if SNAPML_AVAILABLE:
        snapml_pred_proba = snapml_dt.predict_proba(X_test)[:, 1]
        snapml_roc_auc = roc_auc_score(y_test, snapml_pred_proba)
        print(f'[Snap ML Decision Tree] Puntuación ROC-AUC: {snapml_roc_auc:.3f}')
    print("-" * 50)


def compare_svms(X_train, y_train, X_test, y_test):
    """
    Compara el rendimiento de Support Vector Machines (SVMs) de Scikit-learn y Snap ML.

    Args:
        X_train, y_train, X_test, y_test: Conjuntos de datos de entrenamiento y prueba.
    """
    print("\n--- Comparación de Máquinas de Vectores de Soporte (Scikit-learn vs. Snap ML) ---")

    # --- Scikit-learn LinearSVC ---
    sklearn_svm = SklearnLinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
    t0 = time.time()
    sklearn_svm.fit(X_train, y_train)
    sklearn_time = time.time() - t0
    print(f"[Scikit-Learn LinearSVC] Tiempo de entrenamiento (s): {sklearn_time:.2f}")

    # --- Snap ML Support Vector Machine ---
    if SNAPML_AVAILABLE:
        snapml_svm = SnapMLSupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
        print("Parámetros de Snap ML SVM:", snapml_svm.get_params())
        t0 = time.time()
        snapml_svm.fit(X_train, y_train)
        snapml_time = time.time() - t0
        print(f"[Snap ML SupportVectorMachine] Tiempo de entrenamiento (s): {snapml_time:.2f}")

        # Comparación de velocidad
        training_speedup = sklearn_time / snapml_time
        print(f'[Support Vector Machine] Aceleración Snap ML vs. Scikit-Learn: {training_speedup:.2f}x')

    # Evaluación de ROC-AUC (usando decision_function para SVMs)
    sklearn_decision_scores = sklearn_svm.decision_function(X_test)
    sklearn_roc_auc = roc_auc_score(y_test, sklearn_decision_scores)
    print(f"[Scikit-Learn LinearSVC] Puntuación ROC-AUC: {sklearn_roc_auc:.3f}")

    if SNAPML_AVAILABLE:
        snapml_decision_scores = snapml_svm.decision_function(X_test)
        snapml_roc_auc = roc_auc_score(y_test, snapml_decision_scores)
        print(f"[Snap ML SupportVectorMachine] Puntuación ROC-AUC: {snapml_roc_auc:.3f}")

    # Evaluación de Hinge Loss
    loss_sklearn = hinge_loss(y_test, sklearn_decision_scores)
    print(f"[Scikit-Learn LinearSVC] Hinge loss: {loss_sklearn:.3f}")

    if SNAPML_AVAILABLE:
        loss_snapml = hinge_loss(y_test, snapml_decision_scores)
        print(f"[Snap ML SupportVectorMachine] Hinge loss: {loss_snapml:.3f}")
    print("-" * 50)


# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    # --- Configuración del Dataset ---
    DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0RHPEN/data/creditcard.tgz"
    DATA_FILENAME = "creditcard.csv"
    REPLICAS_FOR_INFLATION = 10 # Número de veces para inflar el dataset original

    # 1. Descargar y cargar los datos
    raw_data_df = download_and_load_data(DATA_URL, DATA_FILENAME)

    if raw_data_df.empty:
        print("No se pudo cargar el dataset. Terminando la ejecución.")
        sys.exit(1) # Salir si no hay datos

    # 2. Inflar el dataset
    big_data_df = inflate_dataset(raw_data_df, n_replicas=REPLICAS_FOR_INFLATION)

    # 3. Analizar la distribución de clases y montos
    analyze_class_and_amount_distribution(big_data_df)

    # 4. Preprocesar características
    X_features, y_labels = preprocess_features(big_data_df)

    # 5. Dividir los datos
    X_train, X_test, y_train, y_test = split_data(X_features, y_labels)

    # 6. Comparar Árboles de Decisión (Scikit-learn vs. Snap ML)
    compare_decision_trees(X_train, y_train, X_test, y_test)

    # 7. Comparar Máquinas de Vectores de Soporte (Scikit-learn vs. Snap ML)
    compare_svms(X_train, y_train, X_test, y_test)

    print("\n--- Detección de Fraude Completada ---")
# Detección de Fraude en Tarjetas de Crédito con ML 🪪
Este proyecto explora la aplicación de Machine Learning para la detección de transacciones fraudulentas con tarjetas de crédito. Dado el desbalance inherente en este tipo de datos (muy pocas transacciones son fraudulentas), el proyecto aborda cómo manejar este desequilibrio y compara el rendimiento y la eficiencia de entrenamiento de dos bibliotecas fundamentales: Scikit-learn y Snap ML. El objetivo es identificar transacciones sospechosas de manera rápida y precisa, lo que es crucial para la seguridad financiera.

## Tecnologias usadas 🐍
- pandas: manipulación, limpieza y análisis de datos tabulares (estructuración del dataset de transacciones).
- numpy: operaciones numéricas eficientes, especialmente con arreglos de datos.
- matplotlib: creación de visualizaciones estáticas (histogramas, gráficos de pastel).
- seaborn: creación de visualizaciones estadísticas más atractivas.
- scikit-learn (sklearn): preprocesamiento de datos, balanceo de clases y métricas de evaluación
- snapml: Una biblioteca optimizada de IBM que ofrece algoritmos de ML acelerados por GPU/CPU, utilizada aquí para comparar su velocidad de entrenamiento
- warnings: Para controlar la visualización de advertencias durante la ejecución del script.
- skillsnetwork: Utilizada para la descarga programática del dataset desde una URL proporcionada por IBM.

## Consideraciones en Instalación ⚙️
Si usamos pip:

pip install -q 

  pandas==1.3.4 
   
  numpy==1.21.4 \
  
  seaborn==0.9.0 \
  
  matplotlib==3.5.0 \
  
  scikit-learn==0.20.1

pip install snapml

En esta ocasion el codigo se escribio en Jupyter Notebook para Python, adicionalmente utilizamos la concexion de SkillNetwork (por parte de IBM) para acceder a bases de datos gratuitas.

## Ejemplo de uso 📎
El script realiza un análisis de detección de fraude en transacciones de tarjetas de crédito.
 1. Descarga y Carga de Datos: El dataset creditcard.csv se descarga y se carga en un DataFrame.
 2. Inflado del Dataset: Para simular un escenario de datos más grandes, el dataset original se replica 10 veces.
 3. Análisis de Distribución de Clases: Se muestra un gráfico de pastel de la distribución de transacciones legítimas (Clase 0) y fraudulentas (Clase 1) para ilustrar el desbalance.
 4. Análisis de Montos de Transacción: Se visualiza la distribución de los montos de las transacciones mediante un histograma y se imprimen estadísticas clave (mínimo, máximo, percentil 90).
 5. Preprocesamiento de Datos: Las características V1 a V28 (anonimizadas) se escalan usando StandardScaler.
 6. División de Datos: El dataset se divide en conjuntos de entrenamiento y prueba (70% para entrenamiento, 30% para prueba), manteniendo la proporción de clases (estratificado).
 7. Comparación de Modelos - Árboles de Decisión: Se entrena un DecisionTreeClassifier y se entrena un DecisionTreeClassifier para que se comparen las velocidades de entrenamiento y los puntajes ROC-AUC de ambos modelos.
 8. Comparación de Modelos - Máquinas de Vectores de Soporte (SVM): Se entrena un LinearSVC (SVM lineal) y un SupportVectorMachine para comparar las velocidades de entrenamiento y los puntajes ROC-AUC.
 9. Evaluación de Modelos SVM con Hinge Loss: Se calcula e imprime la métrica hinge_loss para las predicciones de ambos modelos SVM

## Contribuciones 🖨️
Si te interesa contribuir a este proyecto o usarlo independiente, considera:
- Hacer un "fork" del repositorio.
- Crear una nueva rama (git checkout -b feature/nueva-caracteristica).
- Realizar tus cambios y "commitearlos" (git commit -am 'Agregar nueva característica').
- Subir tus cambios a la rama (git push origin feature/nueva-caracteristica).
- Abrir un "Pull Request".

## Licencia 📜
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE (si aplica) para más detalles.


[English Version](README.en.md)

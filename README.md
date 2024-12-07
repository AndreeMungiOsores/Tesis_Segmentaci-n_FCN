# Tesis_Segmentation_FCN
## Segmentación de Imágenes de Tomografía Computarizada para la Detección de Hemorragias Intracraneales
# Descripción del Proyecto
Este proyecto fue desarrollado por Andree Mungi Osores como parte de mi tesis de Ingeniería Mecatrónica. Su propósito es realizar la segmentación de imágenes de tomografía computarizada (CT) para detectar hemorragias intracraneales mediante el uso de redes neuronales convolucionales, específicamente el modelo U-Net.

El proyecto está diseñado para trabajar con un conjunto de datos de imágenes de CT disponibles públicamente, y aplica técnicas de preprocesamiento, generación de datos, entrenamiento y evaluación de modelos para identificar áreas de hemorragia con alta precisión.

#Características Principales
Modelo de Segmentación:

Implementación del modelo U-Net optimizado para la segmentación de imágenes médicas.
Entrenamiento con técnicas avanzadas de augmentación de datos para mejorar la robustez.
Evaluación del Desempeño:

Métricas como el coeficiente de Dice y el índice Jaccard para evaluar la precisión de la segmentación.
Validación cruzada de 5 particiones para garantizar resultados confiables.
Procesamiento de Datos:

Preprocesamiento de imágenes CT mediante normalización y recorte en ventanas específicas.
División de las imágenes en subconjuntos de entrenamiento, validación y prueba.
Estructura del Proyecto
El proyecto está dividido en múltiples módulos para garantizar un desarrollo modular y reutilizable:

data_process.py

Contiene funciones para la normalización de datos, generación de datos de entrenamiento y prueba, y guardado de resultados.
prepare_data.py

Responsable de cargar, preprocesar y dividir las imágenes de CT en subconjuntos de entrenamiento, validación y prueba.
Permite ajustar las imágenes a ventanas específicas para resaltar áreas relevantes del cerebro.
model.py

Implementa la arquitectura U-Net, adaptada para segmentación binaria.
Incluye funciones para calcular la pérdida de Jaccard y compilar el modelo.
main.py

Orquesta el flujo principal del proyecto: preparación de datos, entrenamiento del modelo, y evaluación.
Integra los módulos anteriores y almacena los resultados de segmentación en el disco.
Requisitos Previos
Instalación
Asegúrate de tener instaladas las siguientes bibliotecas:

Python 3.8 o superior
TensorFlow/Keras
NumPy
scikit-learn
nibabel
imageio
pandas
PyQt5 (si deseas integrar la interfaz gráfica)
Puedes instalarlas ejecutando el siguiente comando:

bash
Copiar código
pip install tensorflow numpy scikit-learn nibabel imageio pandas pyqt5
Dataset
Descarga el conjunto de datos de imágenes de CT desde el siguiente enlace:

Dataset de Hemorragias Intracraneales - PhysioNet

Coloca el archivo descargado en el directorio raíz del proyecto y descomprímelo si es necesario.

Instrucciones de Uso
Preparar los Datos: Ejecuta el módulo prepare_data.py para cargar y preprocesar las imágenes de CT:

bash
Copiar código
python prepare_data.py
Entrenar el Modelo: Corre el archivo main.py para entrenar el modelo U-Net con los datos preprocesados:

bash
Copiar código
python main.py
Resultados: Los resultados, incluyendo las segmentaciones y métricas, se guardarán en un directorio llamado results_X, donde X es el número de intento de ejecución.

Visualización de Segmentaciones: Para visualizar las imágenes originales junto con sus máscaras segmentadas, utiliza herramientas como PyQt o scripts adicionales.

Contribuciones
Este proyecto es parte de una tesis académica, por lo que no se aceptan contribuciones externas en este momento. Si tienes alguna pregunta o sugerencia, puedes contactarme directamente.

Licencia
El código y los recursos de este proyecto son de uso exclusivo para fines educativos y de investigación. Para usos comerciales, contacta al autor.

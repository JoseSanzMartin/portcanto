# Proyecto Portcanto

## Descripción
Este proyecto realiza clustering KMeans sobre datos simulados de ciclistas. Se evalúan métricas de homogeneidad, completitud y V-measure, y se lleva un registro de experimentos utilizando MLflow.

---

## **Estructura del proyecto**
- **data/**: Contiene los datasets generados y limpios.
- **model/**: Modelos KMeans entrenados y almacenados.
- **img/**: Imágenes y gráficas generadas.
- **tests/**: Pruebas unitarias del proyecto.
- **scripts/**:
   - `generardataset.py`: Genera el dataset simulado de ciclistas.
   - `clustersciclistes.py`: Realiza el clustering KMeans.
   - `mlflowtracking-K.py`: Ejecuta experimentos con variaciones del parámetro **K** usando MLflow.

---

## **Requisitos**
1. Python 3.8 o superior.
2. Instalación de librerías necesarias:

   ```bash
   pip install -r requirements.txt

## **Ejecución**
1. Genera el dataset con el comando:
python generardataset.py
2. Crea los clusters con el comando:
python clustersciclistes.py

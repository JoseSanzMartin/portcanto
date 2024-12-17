import os
import logging
import numpy as np
import csv

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generar_dataset(num, indice, lista):
    """
    Generamos el dataset de tiempos de los ciclistas

    Args:
        num (int): Número de filas/ciclistas a generar
        indice (int): Índice de inicio de los ciclistas
        lista (list): Lista de diccionarios con los parámetros de generación

    Returns:
        list: Lista de filas con los datos de los ciclistas generados
    """
    data = []
    for i in range(num):
        tipo = lista[i % len(lista)]  # Reparte los tipos cíclicamente
        tiempo_subida = np.random.normal(tipo["mu_p"], tipo["sigma"])
        tiempo_bajada = np.random.normal(tipo["mu_b"], tipo["sigma"])
        tiempo_total = tiempo_subida + tiempo_bajada
        data.append([indice + i, tipo["nombre"],
        round(tiempo_subida, 2),
        round(tiempo_bajada, 2),
        round(tiempo_total, 2)])
    return data

if __name__ == "__main__":

    # Ruta del archivo donde vamos a guardar el dataset
    ARCHIVO_CICLISTAS = 'data/ciclistes.csv'

    # Definición de los tipos de ciclistas y sus medias
    MU_SUBIDA_BUENOS = 3240  # media tiempo subida buenos escaladores
    MU_SUBIDA_MALOS = 4268  # media tiempo subida malos escaladores
    MU_BAJADA_BUENOS = 1440  # media tiempo bajada buenos bajadores
    MU_BAJADA_MALOS = 2160  # media tiempo bajada malos bajadores
    DESVIACION = 240        # desviación estándar: 240 s = 4 min

    # Diccionario con los tipos de ciclistas
    lista_tipos = [
        {"nombre": "BEBB", "mu_p": MU_SUBIDA_BUENOS, "mu_b": MU_BAJADA_BUENOS, "sigma": DESVIACION},
        {"nombre": "BEMB", "mu_p": MU_SUBIDA_BUENOS, "mu_b": MU_BAJADA_MALOS, "sigma": DESVIACION},
        {"nombre": "MEBB", "mu_p": MU_SUBIDA_MALOS, "mu_b": MU_BAJADA_BUENOS, "sigma": DESVIACION},
        {"nombre": "MEMB", "mu_p": MU_SUBIDA_MALOS, "mu_b": MU_BAJADA_MALOS, "sigma": DESVIACION}
    ]

    # Parámetros para generar el dataset
    NUM_CICLISTAS = 100  # Número de ciclistas a generar
    INDICE_INICIAL = 1   # Índice inicial

    # Generamos el dataset
    dataset = generar_dataset(NUM_CICLISTAS, INDICE_INICIAL, lista_tipos)

    # Creamos la carpeta data si no existe
    os.makedirs("data", exist_ok=True)

    # Guardamos el dataset en ciclistas.csv
    with open(ARCHIVO_CICLISTAS, "w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["ID", "TIPO", "SUBIDA (s)", "BAJADA (s)", "TOTAL (s)"])  # Cabecera
        escritor.writerows(dataset)

    print(f"Dataset generado y guardado en {ARCHIVO_CICLISTAS}")
    logging.info("Se ha generado data/ciclistes.csv")

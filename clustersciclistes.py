"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes

    arguments:
        path -- dataset

    Returns: dataframe
    """
    return pd.read_csv(path, delimiter=',')

def EDA(df):
    """
	Exploratory Data Analysis del dataframe

	arguments:
		df -- dataframe

	Returns: None
	"""
    logging.debug('\n%s', df.shape)
    logging.debug('\n%s', df[:5])
    logging.debug('\n%s', df.columns)
    logging.debug('\n%s', df.info())

def clean(df):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

    arguments:
        df -- dataframe

    Returns: dataframe
    """
    df = df.drop(['ID', 'TOTAL (s)'], axis=1)
    logging.debug('\n%s', df[:5])
    return df

def extract_true_labels(df):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)

    arguments:
        df -- dataframe

    Returns: numpy ndarray (true labels)
    """
    # Extraemos la columna 'TIPO' como numpy array
    true_labels = df["TIPO"].to_numpy()
    # Log para ver las primeras etiquetas
    logging.debug('\nEtiquetes verdaderas extraídas:\n%s\n...', true_labels[:5])
    return true_labels

def visualitzar_pairplot(df):
    """
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	arguments:
		df -- dataframe

	Returns: None
	"""
    sns.pairplot(df)
    try:
        os.makedirs(os.path.dirname('img/'))
    except FileExistsError:
        pass
    plt.savefig("img/pairplot.png")

def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	arguments:
		data -- les dades: tp i tb

	Returns: model (objecte KMeans)
	"""
    model = KMeans(n_clusters=4, random_state=42)

    with suppress_stdout_stderr():
        model.fit(data)

    return model

def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. 
    Provem diferents combinacions de parells d'atributs.

    arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades 
        (hem assignat les dades a un dels 4 clústers)

    Returns: None
    """
    try:
        os.makedirs('img/')
    except FileExistsError:
        pass

    # Gràfica 1: SUBIDA (s) vs BAJADA (s)
    fig = plt.figure()
    sns.scatterplot(x='SUBIDA (s)', y='BAJADA (s)', data=data, hue=labels, palette="rainbow")
    plt.title("Clusters: SUBIDA vs BAJADA")
    plt.savefig("img/grafica_subida_bajada.png")
    plt.close()

    # Gràfica 2: SUBIDA (s) vs hora (si existís alguna columna relacionada amb el temps horari)
    # En aquest cas, substitueix-ho per una altra variable existent
    fig = plt.figure()
    sns.scatterplot(x='SUBIDA (s)', y='BAJADA (s)', data=data, hue=labels, palette="rainbow")
    plt.title("Clusters: SUBIDA vs BAJADA (repetició exemple)")
    plt.savefig("img/grafica_subida.png")
    plt.close()

    logging.info("\nS'han generat les gràfiques a la carpeta img/")

def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    # proposta de solució

    dicc = {'tp':0, 'tb': 1}

    logging.info('Centres:')
    for j in range(len(tipus)):
        logging.info(
            '{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(
                j,
                model.cluster_centers_[j][dicc['tp']],
                model.cluster_centers_[j][dicc['tb']]
            )
        )

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(clustering_model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)
    return tipus

def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers,
    i generem 4 fitxers amb les dades dels ciclistes per cadascun dels clústers.

    arguments:
        df -- dataframe (amb la columna 'label')
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    # Crear la carpeta "informes" si no existeix
    try:
        os.makedirs('informes/')
    except FileExistsError:
        pass

    # Generar un fitxer per cada clúster
    for tip in tipus:
        # Crear el nom del fitxer segons el tipus
        fitxer = f"informes/{tip['name'].replace(' ', '_')}.txt"

        # Filtrar el dataframe pel label assignat
        ciclistes = df[df['label'] == tip['label']].index

        # Escriure al fitxer
        with open(fitxer, "w", encoding='utf-8') as foutput:
            for tipus_ciclista in ciclistes:
                foutput.write(f"{tipus_ciclista}\n")  # Escriure cada 'TIPO' al fitxer

        logging.info(f"Fitxer generat: {fitxer}")

    logging.info('\nS\'han generat els informes en la carpeta informes/')


def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers.

    arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model entrenat

    Returns:
        df_nous -- dataframe amb les dades i els clústers assignats
        prediccions -- array de prediccions del model
    """
    # Crear un DataFrame amb les noves dades
    df_nous = pd.DataFrame(dades, columns=['id', 'tp', 'tb', 'tt'])
    
    # Renombrar les columnes per coincidir amb les del model entrenat
    df_nous = df_nous.rename(columns={'tp': 'SUBIDA (s)', 'tb': 'BAJADA (s)'})

    # Seleccionar només les columnes necessàries per fer la predicció
    dades_noves = df_nous[['SUBIDA (s)', 'BAJADA (s)']]

    # Predir els clústers per les noves dades
    prediccions = model.predict(dades_noves)

    # Afegir les prediccions al DataFrame
    df_nous['label'] = prediccions

    logging.info("\nNoves dades amb clústers assignats:\n%s", df_nous)

    return df_nous, prediccions

# ----------------------------------------------

if __name__ == "__main__":

    # Ruta del dataset
    PATH_DATASET = './data/ciclistes.csv'

    """
	TODO:
	load_dataset
	EDA
	clean
	extract_true_labels
	eliminem el tipus, ja no interessa .drop('tipus', axis=1)
	visualitzar_pairplot
	clustering_kmeans
	pickle.dump(...) guardar el model
	mostrar scores i guardar scores
	visualitzar_clusters
	"""

    # Cargamos el dataset
    ciclistes_data = load_dataset(PATH_DATASET)

    # Análisis de los datos
    EDA(ciclistes_data)
    # Limpiamos valores innecesarios
    ciclistes_data = clean(ciclistes_data)

    # Extraemos etiquetas verdaderas
    true_labels = extract_true_labels(ciclistes_data)
    print(ciclistes_data.columns)

    # Extraidas las etiquetas verdaderas, eliminamos tipo
    ciclistes_data = ciclistes_data.drop('TIPO', axis=1)

    # Visualizamos el pairplot
    visualitzar_pairplot(ciclistes_data)

    # Realizamos las tareas de clusterización
    clustering_model = clustering_kmeans(ciclistes_data)

    # guardamos el modelo
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)
    data_labels = clustering_model.labels_

    logging.info('\nHomogeneity: %.3f', homogeneity_score(true_labels, data_labels))
    logging.info('Completeness: %.3f', completeness_score(true_labels, data_labels))
    logging.info('V-measure: %.3f', v_measure_score(true_labels, data_labels))
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump({
	    "h": homogeneity_score(true_labels, data_labels),
	    "c": completeness_score(true_labels, data_labels),
	    "v": v_measure_score(true_labels, data_labels)
	    }, f)

    visualitzar_clusters(ciclistes_data, data_labels)

	# array de diccionaris que assignarà els tipus als labels
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

    """
	afegim la columna label al dataframe
	associar_clusters_patrons(tipus, clustering_model)
	guardem la variable tipus a model/tipus_dict.pkl
	generar_informes
	"""

    ciclistes_data['label'] = clustering_model.labels_.tolist()
    logging.debug('\nColumna label:\n%s', ciclistes_data[:5])

    tipus = associar_clusters_patrons(tipus, clustering_model)
    # guardem la variable tipus
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus, f)
    logging.info('\nTipus i labels:\n%s', tipus)
    print(ciclistes_data.columns)
    # Generació d'informes
    generar_informes(ciclistes_data, tipus)

    # Classificació de nous valors
    nous_ciclistes = [
	    [500, 3230, 1430, 4670], # BEBB
	    [501, 3300, 2120, 5420], # BEMB
	    [502, 4010, 1510, 5520], # MEBB
	    [503, 4350, 2200, 6550] # MEMB
    ]
    
    # Utilitzem la funció nova_prediccio
    df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, clustering_model)

    #Assignació dels nous valors als tipus
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
    
    # Mostrar per pantalla
    print("\nNous ciclistes classificats:")
    print(df_nous_ciclistes)
    # Mostrar los centroides de los clústers
    print("\nCentroides dels clústers:")
    for i, center in enumerate(clustering_model.cluster_centers_):
        print(f"Clúster {i}: SUBIDA (s) = {center[0]:.2f}, BAJADA (s) = {center[1]:.2f}")
    

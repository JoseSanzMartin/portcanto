""" @ IOC - Joan Quintana - 2024 - CE IABD """
import logging
import pickle
import shutil
import mlflow
from mlflow.tracking import MlflowClient

from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

if __name__ == "__main__":

    # Configuració de logging
    logging.basicConfig(format="%(message)s", level=logging.INFO)  # Canviar entre DEBUG i INFO

    # Configuració de l'experiment a MLflow
    client = MlflowClient()
    experiment_name = "K sklearn ciclistes"
    exp = client.get_experiment_by_name(experiment_name)

    if not exp:
        mlflow.create_experiment(
            experiment_name, tags={"mlflow.note.content": "ciclistes variació de paràmetre K"}
        )
        mlflow.set_experiment_tag("version", "1.0")
        mlflow.set_experiment_tag("scikit-learn", "K")
        exp = client.get_experiment_by_name(experiment_name)

    mlflow.set_experiment("K sklearn ciclistes")

    # Funcions auxiliars per eliminar experiments
    def get_run_dir(artifacts_uri):
        """ Retorna la ruta del run """
        return artifacts_uri[7:-10]

    def remove_run_dir(run_dir):
        """ Elimina un path amb shutil.rmtree """
        shutil.rmtree(run_dir, ignore_errors=True)

    # Eliminar tots els runs anteriors
    runs = MlflowClient().search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        mlflow.delete_run(run.info.run_id)
        remove_run_dir(get_run_dir(run.info.artifact_uri))

    # Carreguem i preparem el dataset
    path_dataset = "./data/ciclistes.csv"
    ciclistes_data = load_dataset(path_dataset)
    ciclistes_data_clean = clean(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data_clean)
    ciclistes_data_clean = ciclistes_data_clean.drop("TIPO", axis=1)

    # Llista de valors de K a provar
    Ks = [2, 3, 4, 5, 6, 7, 8]

    # Bucle per provar cada valor de K
    for K in Ks:
        dataset = mlflow.data.from_pandas(ciclistes_data_clean, source=path_dataset)

        # Iniciar un run a MLflow
        mlflow.start_run(description=f"K={K}")
        mlflow.log_input(dataset, context="training")

        # Entrenar el model KMeans
        clustering_model = clustering_kmeans(ciclistes_data_clean, n_clusters=K)
        data_labels = clustering_model.labels_

        # Calcular les mètriques
        h_score = round(homogeneity_score(true_labels, data_labels), 5)
        c_score = round(completeness_score(true_labels, data_labels), 5)
        v_score = round(v_measure_score(true_labels, data_labels), 5)

        # Mostrar resultats al logging
        logging.info(f"K: {K}")
        logging.info(f"H-measure: {h_score:.5f}")
        logging.info(f"C-measure: {c_score:.5f}")
        logging.info(f"V-measure: {v_score:.5f}")

        # Assignar tags i paràmetres
        tags = {
            "engineering": "JQC-IOC",
            "release.candidate": "RC1",
            "release.version": "1.1.2",
        }
        mlflow.set_tags(tags)
        mlflow.log_param("K", K)

        # Registrar mètriques
        mlflow.log_metric("homogeneity", h_score)
        mlflow.log_metric("completeness", c_score)
        mlflow.log_metric("v_measure", v_score)

        # Guardar el model
        model_path = f"./model/kmeans_k{K}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clustering_model, f)
        mlflow.log_artifact(model_path)

        # Guardar el dataset utilitzat
        mlflow.log_artifact(path_dataset)

        # Finalitzar el run
        mlflow.end_run()

    print("S'han generat els runs. Consulteu els resultats a MLflow.")

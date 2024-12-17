"""
@ IOC - CE IABD
"""
import unittest
import os
import pickle

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

class TestGenerarDataset(unittest.TestCase):
    """
    classe TestGenerarDataset
    """
    global mu_p_be
    global mu_p_me
    global mu_b_bb
    global mu_b_mb
    global sigma
    global dicc

    mu_p_be = 3240  # mitjana temps pujada bons escaladors
    mu_p_me = 4268  # mitjana temps pujada mals escaladors
    mu_b_bb = 1440  # mitjana temps baixada bons baixadors
    mu_b_mb = 2160  # mitjana temps baixada mals baixadors
    sigma = 240  # 240 s = 4 min

    dicc = [
        {"nombre": "BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
        {"nombre": "BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
        {"nombre": "MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
        {"nombre": "MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
    ]

    def test_longituddataset(self):
        """
        Test la longitud de l'array
        """
        arr = generar_dataset(200, 1, [dicc[0]])
        self.assertEqual(len(arr), 200)

    def test_valorsmitjatp(self):
        """
        Test del valor mitjà del tp
        """
        arr = generar_dataset(100, 1, [dicc[0]])  # Pasar dicc[0] como lista
        arr_tp = [row[2] for row in arr]  # La columna tp és la tercera (ndice 2)
        tp_mig = sum(arr_tp) / len(arr_tp)
        self.assertLess(tp_mig, 3400)

    def test_valorsmitjatb(self):
        """
        Test del valor mitjà del tb
        """
        arr = generar_dataset(100, 1, [dicc[1]])
        arr_tb = [row[3] for row in arr]  # La columna tb és la cuarta (ndice 3)
        tb_mig = sum(arr_tb) / len(arr_tb)
        self.assertGreater(tb_mig, 2000)

class TestClustersCiclistes(unittest.TestCase):
    """
    classe TestClustersCiclistes
    """
    def setUp(self):
        path_dataset = './data/ciclistes.csv'
        self.ciclistes_data = load_dataset(path_dataset)
        self.ciclistes_data_clean = clean(self.ciclistes_data)
        self.true_labels = extract_true_labels(self.ciclistes_data_clean)
        self.ciclistes_data_clean = self.ciclistes_data_clean.drop('TIPO', axis=1)  # eliminem el tipus, ja no interessa

        self.clustering_model = clustering_kmeans(self.ciclistes_data_clean)
        self.data_labels = self.clustering_model.labels_
        with open('model/clustering_model.pkl', 'wb') as f:
            pickle.dump(self.clustering_model, f)

    def test_check_column(self):
        """
        Comprovem que una columna existeix
        """
        self.assertIn('SUBIDA (s)', self.ciclistes_data_clean.columns)

    def test_data_labels(self):
        """
        Comprovem que data_labels té la mateixa longitud que ciclistes
        """
        self.assertEqual(len(self.data_labels), len(self.ciclistes_data_clean))

    def test_model_saved(self):
        """
        Comprovem que a la carpeta model/ hi ha els fitxer clustering_model.pkl
        """
        check_file = os.path.isfile('./model/clustering_model.pkl')
        self.assertTrue(check_file)

if __name__ == '__main__':
    unittest.main()

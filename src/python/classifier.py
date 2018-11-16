import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools

class Ccomparator:
    def __init__ (self, name, ev):
        self.name = name
        self.ev = ev
    
    def __str__ (self):
        return "{} {}".format(self.name, str(self.ev["accuracy"]))
    
    def __eq__ (self, other):
        return self.ev["accuracy"] == other.ev["accuracy"]
    
    def __lt__ (self, other):
        return self.ev["accuracy"] < other.ev["accuracy"]

# lista de comparadores
comparators = []

# leer archivo con los pacientes
pctes = pd.read_csv("data/weandbnonull.csv")

# vuelvo el label en numeros
pctes["Clase"] = pctes["Clase"].map({"S":0, "F":1, "R":2})

# las columnas, para tener los features
columns = pctes.columns[0:22]
print(columns)

X_train, X_test, y_train, y_test = train_test_split(pctes.iloc[::], pctes["Clase"], test_size=0.2, random_state=42)

genero = tf.feature_column.categorical_column_with_vocabulary_list(
    key="Genero",
    vocabulary_list=["H", "M"])
dpr = tf.feature_column.categorical_column_with_vocabulary_list(
        key="Dpr",
        vocabulary_list=["ICC", "NRL", "PULM", "PABD", "PSTC", "MISC"])
dira = tf.feature_column.categorical_column_with_vocabulary_list(
        key="DIRA",
        vocabulary_list=["ICC", "NRL", "PULM", "PABD", "MISC"])
cDira = tf.feature_column.categorical_column_with_vocabulary_list(
        key="CausaDIRA",
        vocabulary_list=["PULM1", "PULM2", "NRL"])
modo = tf.feature_column.categorical_column_with_vocabulary_list(
        key="Modo",
        vocabulary_list=["AC", "PS"])

feature_columns = [
    tf.feature_column.indicator_column(genero),
    tf.feature_column.numeric_column(key="Edad"),
    tf.feature_column.indicator_column(dpr),
    tf.feature_column.indicator_column(dira),
    tf.feature_column.indicator_column(cDira),
    tf.feature_column.indicator_column(modo),
    tf.feature_column.numeric_column(key="TAS_antes"),
    tf.feature_column.numeric_column(key="TAD_antes"),
    tf.feature_column.numeric_column(key="FC_antes"),
    tf.feature_column.numeric_column(key="TAS"),
    tf.feature_column.numeric_column(key="TAD"),
    tf.feature_column.numeric_column(key="FC"),
    tf.feature_column.numeric_column(key="FR_antes"),
    tf.feature_column.numeric_column(key="FiO2_antes"),
    tf.feature_column.numeric_column(key="FiO2"),
    tf.feature_column.numeric_column(key="PEEP"),
    tf.feature_column.numeric_column(key="Temp"),
    tf.feature_column.numeric_column(key="dVM"),
    tf.feature_column.numeric_column(key="Hb")]

def input_fn (df, labels):
    feature_cols = {k:tf.constant(df[k].values, shape=[df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape=[labels.size, 1])
    return feature_cols, label

# itera capas y numero de neuronas por capa
def DNNiter (minCapas, maxCapas, minNeuronas, maxNeuronas, steps):
    classifier = None # variable que alberga el clasificador
    ev = None # variable que alberga la evaluacion

    # genero wl arrglo con el numero de neuronas
    neuronas = []
    for i in range(minNeuronas,maxNeuronas+1):
        neuronas.append(i)
    
    # genero las capas
    for n in range(minCapas, maxCapas+1):
        for combi in itertools.product(neuronas, repeat=n):
            classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=list(combi), n_classes = 3)
            classifier.train(input_fn=lambda: input_fn(X_train,y_train), steps=steps)
            ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test), steps=1)
            comparators.append(Ccomparator("DNN{}".format(str(list(combi))), ev))
            print("DNN{}".format(str(list(combi))), ev)

# imprime la lista de comparadores a un archivo
def print_comparators_file (path):
    file = open(path, "w+")
    comparators.sort()
    for ccomp in comparators:
        file.write(str(ccomp) + "\n")
    file.close()

DNNiter(2,4,5,10,500)
print_comparators_file("accuracy2.txt")
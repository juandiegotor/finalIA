import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools
import sys

# leer archivo con los pacientes
pctes = pd.read_csv("data/weandbnore.csv")

# vuelvo el label en numeros
pctes["Clase"] = pctes["Clase"].map({"S":0, "F":1})

# las columnas, para tener los features
columns = pctes.columns[0:22]

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

def createDNN (hidden_units, steps):
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=hidden_units, n_classes = 2, model_dir="./grafo/grafo0/")
    classifier.train(input_fn=lambda: input_fn(X_train,y_train), steps=steps)
    ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test), steps=1)
    return ev, classifier

def appendEvaluation (path, ev, hidden_units):
    file = open(path, "a")
    file.write("DNN{}: {}\n".format(str(hidden_units), str(ev)))
    file.close()

hidden_units = []
# Leo parametros y evaluo
for i in range(1, len(sys.argv)):
    hidden_units.append(int(sys.argv[i]))

ev, classifier = createDNN(hidden_units, 1000)
appendEvaluation("./output/python/salida2.txt", ev, hidden_units)
print ev
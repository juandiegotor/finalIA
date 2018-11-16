import tensorflow as tf
import pandas as pd

dfile = "data/weandb01.csv"

pctes = pd.read_csv(dfile)
feature_names = pctes.columns[0:19]
print(feature_names)

# las columnas definiciones y que tales
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

record_defaults = [
    tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.string),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.float32),
    tf.constant([], dtype=tf.int32)]


def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
       parsed_line = tf.decode_csv(line, record_defaults=record_defaults)
       label = parsed_line[-1:] # Last element is the label
       del parsed_line[-1] # Delete last element
       features = parsed_line # Everything (but last element) are the features
       d = dict(zip(feature_names, features)), label
       return d

    dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def serving_input_receiver_fn():
    inputs = {
        'Genero': tf.constant([], dtype=tf.string),
        'Edad': tf.constant([], dtype=tf.float32),
        'Dpr': tf.constant([], dtype=tf.string),
        'DIRA': tf.constant([], dtype=tf.string),
        'CausaDIRA': tf.constant([], dtype=tf.string),
        'Modo': tf.constant([], dtype=tf.string),
        'TAS_antes': tf.constant([], dtype=tf.float32),
        'TAD_antes': tf.constant([], dtype=tf.float32),
        'FC_antes': tf.constant([], dtype=tf.float32),
        'TAS': tf.constant([], dtype=tf.float32),
        'TAD': tf.constant([], dtype=tf.float32),
        'FC': tf.constant([], dtype=tf.float32),
        'FR_antes': tf.constant([], dtype=tf.float32),
        'FiO2_antes': tf.constant([], dtype=tf.float32),
        'FiO2': tf.constant([], dtype=tf.float32),
        'PEEP': tf.constant([], dtype=tf.float32),
        'Temp': tf.constant([], dtype=tf.float32),
        'dVM': tf.constant([], dtype=tf.float32),
        'Hb': tf.constant([], dtype=tf.float32)
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[3],
    n_classes=2)

classifier.train(
    input_fn=lambda: my_input_fn(dfile, True, 8))

evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn(dfile, False, 4))

print("evaluate results")
for key in evaluate_result:
    print(" {}, was: {}".format(key, evaluate_result[key]))

predict_results = classifier.predict(
    input_fn=lambda: my_input_fn(dfile, False, 1))

print("Predictions")
predictions = []
for prediction in predict_results:
    predictions.append(prediction["class_ids"][0])

confusion_matrix = tf.confusion_matrix(pctes["Clase"].tolist(), predictions)
with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))

import pickle, sys
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import keras_metrics
import keras.backend as K

from scipy.sparse import lil_matrix

from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Flatten, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert.bert import *

from bert import tokenization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support

from ncbi_normalization import vectorizer
from ncbi_normalization import load
from ncbi_normalization import sample


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set parameters:
batch_size = 10
max_batch_size = 10
epochs = 10
maxlen = 384


def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_data(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)

def load_concepts():
    # MEDIC dictionary
    dictionary = load.Terminology()
    # dictionary of entries, key = canonical id, value = named tuple in the form of
    #   MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
    #   AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
    dictionary.loaded = load.load('data/CTD_diseases.tsv','MEDIC')

    from ncbi_normalization.parse_MEDIC_dictionary import concept_obj
    print('Using truncated development corpus for evaluation.')
    #logger.info('Using truncated development corpus for evaluation.')
    [real_val_data,concept_order,corpus_dev] = pickle.load(open('data/gitig_real_val_data_truncated_d50_p5.pickle','rb'))
    real_val_data.y=np.array(real_val_data.y)
    concept = concept_obj(dictionary,order=concept_order)

    return concept, real_val_data

def load_mentions():
    pass

def tokenize(abstracts, maxlen=512):
    tokenizer = tokenization.FullTokenizer("biobert_v1.1_pubmed/vocab.txt", do_lower_case=False)
    

    ret_val = []
    for abstract in tqdm(abstracts, desc="Tokenizing abstracts"):
        abstract = ["[CLS]"] + tokenizer.tokenize(abstract[0:maxlen-2]) + ["[SEP]"]
        ret_val.append(abstract)
    return ret_val, tokenizer.vocab


def transform(abstracts_file, mesh_file):

    print("Reading input files..")

    abstracts = load_data("./" + abstracts_file)
    labels = load_data("./" + mesh_file)

    abstracts, vocab = tokenize(abstracts, maxlen=maxlen)

    print("Vectorizing..")
    token_vectors = np.asarray( [np.asarray( [vocab[token] for token in abstract] + [0] * (maxlen - len(abstract)) ) for abstract in abstracts] )
    del abstracts
    print("Token_vectors shape:", token_vectors.shape)

    print("Binarizing labels..")
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    labels = one_hot_encoder.fit_transform(np.asarray(labels).reshape(-1,1))
    labels = labels.astype('b')
    print("Labels shape:", labels.shape)
    #print(np.dtype(labels))

    print("Splitting..")
    token_vectors_train, token_vectors_test, labels_train, labels_test = train_test_split(token_vectors, labels, test_size=0.1)

    _, sequence_len = token_vectors_train.shape

    return token_vectors_train, token_vectors_test, labels_train, labels_test, sequence_len


def build_model(abstracts_train, abstracts_test, labels_train, labels_test, sequence_len):

    checkpoint_file = "../../biobert_pubmed/biobert_model.ckpt"
    config_file = "../../biobert_pubmed/bert_config.json"

    biobert = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False, seq_len=sequence_len)
    #biobert_train = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=sequence_len)

    # Unfreeze bert layers.
    for layer in biobert.layers[:]:
        layer.trainable = True

    print(biobert.input)
    print(biobert.layers[-1].output)

    print(tf.slice(biobert.layers[-1].output, [0, 0, 0], [-1, 1, -1]))

    slice_layer = Lambda(lambda x: tf.slice(x, [0, 0, 0], [-1, 1, -1]))(biobert.layers[-1].output)

    flatten_layer = Flatten()(slice_layer)

    output_layer = Dense(labels_train.shape[1], activation='softmax')(flatten_layer)

    model = Model(biobert.input, output_layer)

    print(model.summary(line_length=118))

    learning_rate = 0.00005

    model.compile(loss='categorical_crossentropy',
                metrics=[keras_metrics.precision(), keras_metrics.recall()],
                optimizer=Adam(lr=learning_rate))#SGD(lr=0.2, momentum=0.9))

    best_f1 = 0.0
    stale_epochs = 0

    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        #learning_rate -= 0.0001
        cur_batch_size = min(batch_size + int(0.125 * epoch * batch_size), max_batch_size)
        print("batch size:", cur_batch_size)
        # model.compile(loss='binary_crossentropy',
        #         optimizer=Adam(lr=learning_rate))
        print("learning rate:", K.eval(model.optimizer.lr))
        model_hist = model.fit([abstracts_train, np.zeros_like(abstracts_train)], labels_train,
            batch_size=cur_batch_size,
            epochs=1,
            validation_data=[[abstracts_test, np.zeros_like(abstracts_test)], labels_test])

        precision = model_hist.history['val_precision'][0]
        recall = model_hist.history['val_recall'][0]
        f1 = (2.0 * precision * recall) / (precision + recall)

        print("F1-score:", f1, "\n")

        if f1 > best_f1:
            best_f1 = f1
            stale_epochs = 0
            print("Saving model..\n")
            model.save(sys.argv[3])
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break

build_model(*transform(sys.argv[1], sys.argv[2]))
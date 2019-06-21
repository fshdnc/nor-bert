'''
no context
'''

import pickle, sys, os, random, time, math
random.seed(1)
import logging
import logging.config
import configparser as cp
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
from keras.preprocessing.sequence import pad_sequences

from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert.bert import *

from bert import tokenization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support

from ncbi_normalization import load
from ncbi_normalization.parse_MEDIC_dictionary import concept_obj
from ncbi_normalization import vectorizer, sample, callback

#configurations
TIME = time.strftime('%Y%m%d-%H%M%S')
dynamic_defaults = {'timestamp': TIME}
config = cp.ConfigParser(defaults=dynamic_defaults,interpolation=cp.ExtendedInterpolation(),strict=False)
try:
    directory = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    config.read(os.path.join(directory, 'src/defaults.cfg'))
except NameError:
    directory = '/home/lhchan/nor-bert/src'
    config.read(os.path.join(directory, 'defaults.cfg'))

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
set_session(tf.Session(config=gpu_config))

#logging
logger = logging.getLogger(__name__)
from src.settings import LOGGING_SETTINGS
logging.config.dictConfig(LOGGING_SETTINGS)


def dump_data(file_name, data):

    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_data(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)

def load_concepts(dict_file):
    '''
    dict_file: directory to the tsv file of MEDIC dictionary
    dictionary.loaded format:
        dictionary of entries, key = canonical id, value = named tuple in the form of
        MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
        AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
    '''
    # MEDIC dictionary
    dictionary = load.Terminology()
    dictionary.loaded = load.load(dict_file,'MEDIC')

    logger.info('Using sampled development corpus for evaluation.')
    smpl_dev_data = sample.Data()
    corpus_dev_sampled = sample.NewDataSet('dev corpus')
    [smpl_dev_data.mentions,smpl_dev_data.y,concept_order,corpus_dev_sampled.ids,corpus_dev_sampled.info,corpus_dev_sampled.names] = load_data('data/sampled_dev_set.pickle')
    smpl_dev_data.y=np.array(smpl_dev_data.y)
    concept = concept_obj(dictionary,order=concept_order)
    concept.names = [name.lower() for name in concept.names]

    return concept, smpl_dev_data, dictionary, corpus_dev_sampled


def load_mentions(corpus_file,corpus_type):
    '''
    corpus_file: file to the ncbi set
    corpus_type: 'training corpus' or 'dev corpus' or 'test corpus'
    '''
    corpus = sample.NewDataSet(corpus_type)
    corpus.objects = load.load(corpus_file,'NCBI')

    corpus.ids = [] # list of all ids (gold standard for each mention)
    corpus.names = [] # list of all names
    corpus.all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                corpus.ids.append(nor_ids) # append list of ids, usually len(list)=1
                corpus.names.append(mention.text)
                corpus.all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))
    corpus.names = [name.lower() for name in corpus.names]

    return corpus


def transform(tokenizer,mention,concepts,mmaxlen,cmaxlen):
    '''
    Takes in one mention, 1 pos and n neg concepts
    '''
    mention = tokenizer.tokenize(mention)[:mmaxlen]
    # NO LOWERCASE
    instances = [["[CLS]"] + mention + ["[SEP]"] + tokenizer.tokenize(concept)[:cmaxlen] + ["[SEP]"] for concept in concepts]
    segmentation = [[0]*(len(mention)+2)+[1]*(len(concept)+1) for concept in concepts]
    instances = np.asarray([np.asarray([tokenizer.vocab[token] for token in instance] + [0] * (mmaxlen + cmaxlen + 3 - len(instance)) ) for instance in instances])
    segmentation = pad_sequences(np.array(segmentation),padding='post',maxlen=mmaxlen+cmaxlen+3)
    # sanity check
    _, sequence_len = instances.shape
    assert sequence_len == mmaxlen + cmaxlen + 3

    return instances, segmentation


def examples(concept, positives, tokenizer, neg_count, mmaxlen, cmaxlen):
    """
    Builds positive and negative examples.
    """
    while True:
        for (chosen_idx, idces), m_span in positives:          
            if len(chosen_idx) ==1:
                # FIXME: only taking into account whose gold standard has exactly one concept
                c_span = [concept.names[chosen_idx[0]]]
                negative_spans = [concept.names[i] for i in random.sample(list(set([*range(len(concept.names))])-set(idces)),neg_count)]

                inputs, segmentation = transform(tokenizer,m_span,c_span+negative_spans,mmaxlen,cmaxlen)
                distances = [1] + [0]*neg_count
                data = {
                    'Input-Token': inputs,
                    'Input-Segment': segmentation,
                    'prediction_layer': np.asarray(distances),
                }
                yield data, data


def transform_concepts(concept, tokenizer, cmaxlen):
    '''
    Takes in concept object
    '''
    concept = [tokenizer.tokenize(span)[:cmaxlen] for span in concept.names]
    return concept

def examples_prediction(vectorized_concepts, positives, mmaxlen, cmaxlen, batch_size):
    while True:
        for (chosen_idx, idces), m_span in tqdm(positives, ascii=True, desc='Predicting'):
            # FIXME: one for loop takes forever
            mention = tokenizer.tokenize(m_span)[:mmaxlen]
            instances = [["[CLS]"] + mention + ["[SEP]"] + concept + ["[SEP]"] for concept in vectorized_concepts]
            instances = np.asarray([np.asarray([tokenizer.vocab[token] for token in instance] + [0] * (mmaxlen + cmaxlen + 3 - len(instance)) ) for instance in instances])

            segmentation = [[0]*(len(mention)+2)+[1]*(len(concept)+1) for concept in vectorized_concepts]
            segmentation = pad_sequences(np.array(segmentation),padding='post',maxlen=mmaxlen+cmaxlen+3)

            distances = [0] # dummy

            for batch in range(math.ceil(len(instances)/batch_size)):
                frag_instances = instances[batch*batch_size:(batch+1)*batch_size]
                frag_segmentation = segmentation[batch*batch_size:(batch+1)*batch_size]

                data = {
                    'Input-Token': frag_instances,
                    'Input-Segment': frag_segmentation,
                    'prediction_layer': np.asarray(distances),
                }

                yield data, data


from datetime import datetime
from keras.callbacks import Callback

class EarlyStoppingRankingAccuracyGenerator(Callback):
    '''
    Ranking accuracy callback with early stopping.
    '''
    def __init__(self, conf, original_model, concept, val_generator, val_data, prediction_steps):
        super().__init__()
        self.conf = conf
        self.original_model = original_model
        self.concept = concept
        self.val_generator = val_generator
        self.val_data = val_data
        self.prediction_steps = prediction_steps

        self.best = 0 # best accuracy
        self.wait = 0
        self.stopped_epoch = 0
        self.patience = int(conf['training']['patience'])
        self.model_path = conf['model']['path_model_whole']

        #self.save = int(self.conf['settings']['save_prediction'])
        self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.history = self.conf['settings']['history'] + self.now + '.txt'
        callback.write_training_info(self.conf,self.history)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

        self.wait = 0
        with open(self.history,'a',encoding='utf-8') as fh:
        # Pass the file handle in as a lambda function to make it callable
            self.original_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        predictions = self.original_model.predict_generator(self.val_generator,steps=self.prediction_steps)
        evaluation_parameter = callback.evaluate(self.val_data.mentions, predictions, self.val_data.y)
        self.accuracy.append(evaluation_parameter)

        with open(self.history,'a',encoding='utf-8') as f:
            f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))
            if logs.get('val_loss'):
                f.write('Epoch: {0}, Validation loss: {1}\n'.format(epoch,logs.get('val_loss')))
                
        if evaluation_parameter > self.best:
            logging.info('Intermediate model saved.')
            self.best = evaluation_parameter
            self.original_model.save_weights(self.model_path)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > int(self.conf['training']['patience']):
                self.stopped_epoch = epoch
                self.original_model.stop_training = True
        logger.info('Testing: epoch: {0}, self.original_model.stop_training: {1}'.format(epoch,self.original_model.stop_training))
        return

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logger.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
        try:
            self.original_model.load_weights(self.model_path)
            logger.info('Best model reloaded.')
        except OSError:
            logger.info('No saved model found.')
        predictions = self.original_model.predict_generator(self.val_generator,steps=self.prediction_steps)
        evaluation_parameter = callback.evaluate(self.val_data.mentions, predictions, self.val_data.y,write=self.history,concept=self.concept)
        if not self.conf.getint('model','save'): # if don't want to save model, delete the cached model
            try:
                os.remove(self.model_path)
            except FileNotFoundError:
                pass
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return


def build_model(checkpoint_file,config_file,sequence_len,learning_rate):
    biobert = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False, seq_len=sequence_len)
    #biobert_train = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=sequence_len)

    # Unfreeze bert layers.
    for layer in biobert.layers[:]:
        layer.trainable = True

    logger.info(biobert.input)
    logger.info(biobert.layers[-1].output)

    logger.info(tf.slice(biobert.layers[-1].output, [0, 0, 0], [-1, 1, -1]))

    slice_layer = Lambda(lambda x: tf.slice(x, [0, 0, 0], [-1, 1, -1]))(biobert.layers[-1].output)

    flatten_layer = Flatten()(slice_layer)

    prediction_layer = Dense(1, activation='sigmoid',name='prediction_layer')(flatten_layer)

    model = Model(inputs=biobert.input, outputs=prediction_layer)

    logger.info(model.summary(line_length=118))

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=learning_rate))#SGD(lr=0.2, momentum=0.9))

    return model



if __name__ == "__main__":

    # concepts
    concept, smpl_dev_data, dictionary, corpus_dev_sampled = load_concepts(config['terminology']['dict_file'])
    # mentions
    corpus_train = load_mentions(config['corpus']['training_file'],'training corpus')
    corpus_dev = load_mentions(config['corpus']['development_file'],'dev corpus')

    tokenizer = tokenization.FullTokenizer(config['bert']['vocab_file'], do_lower_case=False)

    # FIXME: only using one concept name per mention
    positives_training, positives_dev, positives_dev_sampled = load_data('data/gitig_positive_indices.pickle')
    del positives_dev
    positives_training = [(_, span.lower()) for _, span in positives_training]
    positives_dev_sampled = [(_, span.lower()) for _, span in positives_dev_sampled]

    # generators for training and validation instances
    train_examples = examples(concept,positives_training,tokenizer,config.getint('training','neg_count'),config.getint('training','mmaxlen'),config.getint('training','cmaxlen'))
    dev_examples = examples(concept,positives_dev_sampled,tokenizer,config.getint('training','neg_count'),config.getint('training','mmaxlen'),config.getint('training','cmaxlen'))
    concept.biobert = transform_concepts(concept, tokenizer, cmaxlen = config.getint('training','cmaxlen'))
    prediction_examples = examples_prediction(concept.biobert, positives_dev_sampled, config.getint('training','mmaxlen'),config.getint('training','cmaxlen'),config.getint('training','batch_size'))

    # steps for fit_generator
    train_steps = len([0 for (chosen_idx, idces), m_span in positives_training if len(chosen_idx) ==1])
    dev_steps = len(smpl_dev_data.mentions)
    seq_len = config.getint('training','mmaxlen') + config.getint('training','cmaxlen') + 3

    model = build_model(config['bert']['checkpoint_file'],config['bert']['config_file'],seq_len,config.getfloat('model','lr'))
    eval_function = EarlyStoppingRankingAccuracyGenerator(config, model, concept, prediction_examples, smpl_dev_data, math.ceil(len(concept.names)/config.getint('training','batch_size'))*len(smpl_dev_data.mentions))
    hist = model.fit_generator(train_examples, steps_per_epoch=train_steps,validation_data=dev_examples, validation_steps=dev_steps, epochs=config.getint('training','epochs'), callbacks=[eval_function])



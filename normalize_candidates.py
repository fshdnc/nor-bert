'''
use candidates to reduce prediction time
'''

import random
random.seed(1)
import logging
import logging.config
import configparser as cp
import tensorflow as tf
from tqdm import tqdm

from bert import tokenization

from normalize import *
from ncbi_normalization import sample

#logging
logger = logging.getLogger(__name__)
from src.settings import LOGGING_SETTINGS
logging.config.dictConfig(LOGGING_SETTINGS)


def examples_candidate(candidates, positives, tokenizer, concept, mmaxlen, cmaxlen):
    assert len(candidates) == len(positives)
    while True:
        for cans, poss in tqdm(zip(candidates,positives), ascii=True, desc='Predicting'):
            (chosen_idx, idces), m_span = poss
            mention = tokenizer.tokenize(m_span)[:mmaxlen]

            inputs, segmentation = transform(tokenizer,m_span,concept[cans],mmaxlen,cmaxlen)
            distances = [0] # dummy
            data = {
                'Input-Token': inputs,
                'Input-Segment': segmentation,
                'prediction_layer': np.asarray(distances),
            }

            yield data, data


if __name__ == "__main__":
	
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


	# concepts
	concept, smpl_dev_data, dictionary, corpus_dev_sampled = load_concepts(config['terminology']['dict_file'])
	# mentions
	corpus_train = load_mentions(config['corpus']['training_file'],'training corpus')
	corpus_dev = load_mentions(config['corpus']['development_file'],'dev corpus')

	tokenizer = tokenization.FullTokenizer(config['bert']['vocab_file'], do_lower_case=False)

	# FIXME: only using one concept name per mention
	positives_training, positives_dev, positives_dev_sampled = load_data('data/gitig_positive_indices.pickle')
	positives_training = [(_, span.lower()) for _, span in positives_training]
	positives_dev = [(_, span.lower()) for _, span in positives_dev]
	positives_dev_sampled = [(_, span.lower()) for _, span in positives_dev_sampled]

	# candidates
	can_val_data = sample.NewDataSet('dev corpus')
	[candidates, can_val_data.mentions, can_val_data.y] = load_data('data/gitig_selected_19.pickle')
	can_val_data.y = np.array(can_val_data.y)

	# generators for training and validation instances
	train_examples = examples(concept,positives_training,tokenizer,config.getint('training','neg_count'),config.getint('training','mmaxlen'),config.getint('training','cmaxlen'))
	dev_examples = examples(concept,positives_dev_sampled,tokenizer,config.getint('training','neg_count'),config.getint('training','mmaxlen'),config.getint('training','cmaxlen'))
	concept.biobert = transform_concepts(concept, tokenizer, cmaxlen = config.getint('training','cmaxlen'))
	prediction_examples = examples_candidate(candidates, positives_dev, tokenizer, np.array(concept.names), config.getint('training','mmaxlen'),config.getint('training','cmaxlen'))


	# steps for fit_generator
	train_steps = len([0 for (chosen_idx, idces), m_span in positives_training if len(chosen_idx) ==1])
	dev_steps = len(smpl_dev_data.mentions)
	seq_len = config.getint('training','mmaxlen') + config.getint('training','cmaxlen') + 3

	model = build_model(config['bert']['checkpoint_file'],config['bert']['config_file'],seq_len,config.getfloat('model','lr'))
	eval_function = EarlyStoppingRankingAccuracyGenerator(config, model, concept, prediction_examples, can_val_data, len(can_val_data.mentions))
	hist = model.fit_generator(train_examples, steps_per_epoch=train_steps,validation_data=dev_examples, validation_steps=dev_steps, epochs=config.getint('training','epochs'), callbacks=[eval_function])

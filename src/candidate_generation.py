'''
candidate generation: writes a pickle file of candidates

'''

import sys
import nltk
import numpy as np

from ncbi_normalization import load, sample
from ncbi_normalization.parse_MEDIC_dictionary import concept_obj

from normalize import dump_data, load_data, load_mentions


from gensim.models import KeyedVectors

def prepare_embedding_vocab(filename, binary = True, limit = 1000000):
    '''filename: '~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin'
       1. Use gensim for reading in embedding model
       2. Sort based on the index to make sure that they are in the correct order
       3. Normalize the vectors
       4. Build vocabulary mappings, zero for padding
       5. Create an inverse dictionary
    '''
    vector_model = KeyedVectors.load_word2vec_format(filename, binary = binary, limit = limit)
    #vector_model=KeyedVectors.load_word2vec_format(config['embedding']['emb_file'], binary=True, limit=50000)
    words = [k for k,v in sorted(vector_model.vocab.items(),key = lambda x:x[1].index)]
    vector_model.init_sims(replace = True)
    vocabulary={"<SPECIAL>": 0, "<OOV>": 1}
    for word in words:
        vocabulary.setdefault(word, len(vocabulary))
    inversed_vocabulary={value:key for key, value in vocabulary.items()}
    return vector_model, vocabulary, inversed_vocabulary

def load_pretrained_word_embeddings(vocab,embedding_model):
    """vocab: vocabulary from data vectorizer
       embedding_model: model loaded with gensim"""
    pretrained_embeddings = np.random.uniform(low=-0.05, high=0.05, size=(len(vocab)-1,embedding_model.vectors.shape[1]))
    pretrained_embeddings = np.vstack((np.zeros(shape=(1,embedding_model.vectors.shape[1])), pretrained_embeddings))
    found=0
    for word,idx in vocab.items():
        if word in embedding_model.vocab:
            pretrained_embeddings[idx]=embedding_model.get_vector(word)
            found+=1           
    print("Found pretrained vectors for {found} words.".format(found=found))
    return pretrained_embeddings

def load_concepts(dict_file,order):
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

    concept = concept_obj(dictionary,order=order)
    concept.names = [name.lower() for name in concept.names]

    return concept, dictionary

def span_to_sum_of_w2v(spans,vocabulary,pretrained):
    '''
    represent all spans by sum of w2v
    '''
    embeddings = []
    for span in spans:
        tokenized = nltk.word_tokenize(span.lower())
        index = [vocabulary.get(token,1) for token in tokenized]
        #emb = np.mean(np.array([pretrained[i] for i in index]), axis=0)
        emb = np.sum(np.array([pretrained[i] for i in index]), axis=0)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    return embeddings

def cosine_similarity_candidates(mention_spans,concept_spans,emb_path,n_cossim):
    '''
    yields list of list of candidates

    n_cossim = number of candidates for each mention
    '''
    # prepare embeddings
    vector_model, vocabulary, inversed_vocabulary = prepare_embedding_vocab(emb_path, binary = True)
    pretrained = load_pretrained_word_embeddings(vocabulary, vector_model)

    # vector representations
    mention_embeddings = span_to_sum_of_w2v(mention_spans,vocabulary,pretrained)
    concept_embeddings = span_to_sum_of_w2v(concept_spans,vocabulary,pretrained)

    from sklearn.preprocessing import normalize
    concept_embeddings = normalize(concept_embeddings)
    mention_embeddings = normalize(mention_embeddings)

    dot_product_matrix = np.dot(mention_embeddings,np.transpose(concept_embeddings))
    dot_product_matrix = dot_product_matrix.tolist()
    candidate_indices = [np.argpartition(np.array(mention_candidates),-n_cossim)[-n_cossim:].tolist() for mention_candidates in dot_product_matrix]

    return candidate_indices

def jaccard_distance_candidates(mention_spans,concept_spans,n_jaccard):
    candidate_indices = []
    for mention in mention_spans:
        distances = [nltk.jaccard_distance(set(mention),set(concept)) for concept in concept_spans]
        indices = np.argpartition(np.array(distances),-n_jaccard)[-n_jaccard:].tolist()
        candidate_indices.append(indices)
    return candidate_indices

if __name__ == "__main__":
    '''
    1. prepare concept spans & mention spans
    2. get the candidates based on cosine similarity
    3. get the candidates based on Jaccard distance
    4. prepare (start, end, span), gold standard
    '''

    dict_file = 'data/CTD_diseases.tsv'
    dev_file = 'data/NCBIdevelopset_corpus.txt'
    emb_path = 'data/wvec_50_haodi-li-et-al.bin'
    n_cossim = sys.argv[1]
    n_jaccard = sys.argv[2]
    save_to = 'data/selected_max200.pickle'

    # (1)
    # concepts
    [potato0,potato1,concept_order,potato2,potato3,potato4] = load_data('data/sampled_dev_set.pickle')
    del potato0, potato1, potato2, potato3, potato4
    concept, dictionary = load_concepts(dict_file,concept_order)

    # mentions
    corpus_dev = load_mentions(dev_file,'dev corpus')

    # (2)
    cossim_candidate_indices = cosine_similarity_candidates(corpus_dev.names,concept.names,emb_path,n_cossim)

    # (3)
    jaccard_candidate_indices = jaccard_distance_candidates(corpus_dev.names,concept.names,n_jaccard)

    # (4)
    assert len(cossim_candidate_indices)==len(jaccard_candidate_indices)
    candidates = []
    for cossim,jaccard in zip(cossim_candidate_indices,jaccard_candidate_indices):
        mention_candidates = sorted(list(set(cossim+jaccard)))
        candidates.append(mention_candidates)

    positives_training, positives_dev, positives_dev_truncated = load_data('data/gitig_positive_indices.pickle')
    del positives_training, positives_dev_truncated
    positives_dev = sample.prepare_positives(positives_dev,nltk.word_tokenize,vocabulary)

    can_val_data = sample.NewDataSet('dev corpus')
    can_val_data.y = []
    can_val_data.mentions = []
    start = 0
    for cans, poss, span in zip(candidates,positives_dev,corpus_dev.names):
        end = start + len(cans)
        (chosen_idx, idces), e_token_indices = poss
        can_val_data.y.extend([1 if can in idces else 0 for can in cans])
        can_val_data.mentions.append((start,end,span))
        start = end

    assert len(can_val_data.mentions)==len(candidates)

    data = [candidates, can_val_data.mentions, can_val_data.y]
    dump_data(save_to,data)




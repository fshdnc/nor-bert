#!/usr/bin7env python3
# coding:utf8

from collections import namedtuple

'''read in MEDIC terminology, terminology file from DNorm system
a lot of files in DNorm-0.0.7/data not looked into

read in MEDIC terminology into namedtuples
read-in format:
	DiseaseName
	DiseaseID
	AltDiseaseIDs
	Definition (usually none)
	ParentIDs
	TreeNumbers
	ParentTreeNumbers
	Synonyms

returned format:
namedtuple(
	DiseaseID
	DiseaseName
	AllDiseaseIDs: DiseaseID + AltDiseaseIDs
	AllNames: DiseaseName + Synonyms

'''

MEDIC_ENTRY = namedtuple('MEDIC_ENTRY','DiseaseID DiseaseName AllDiseaseIDs AllNames Def')

#namedtuple: https://stackoverflow.com/questions/2970608/what-are-named-tuples-in-pythons
def parse_MEDIC_dictionary(filename):
	with open(filename,'r') as f:
		for line in f:
			if not line.startswith("#"):
				DiseaseName, DiseaseID, AltDiseaseIDs, Def, _, _, _, Synonyms = line.strip('\n').split('\t')
				AllDiseaseIDs = tuple([DiseaseID]+AltDiseaseIDs.split('|')) if AltDiseaseIDs else tuple([DiseaseID])
				AllNames = tuple([DiseaseName]+Synonyms.split('|')) if Synonyms else tuple([DiseaseName])
				entry = MEDIC_ENTRY(DiseaseID,DiseaseName,AllDiseaseIDs,AllNames,Def)
				yield DiseaseID, entry

#dunno what will happend if no altID or syn
#some AllNames tuples have comma at the end but does not seem to affect the functionality

def parse_MEDIC_dictionary_newer(filename):
	with open(filename,'r') as f:
		for line in f:
			if not line.startswith("#"):
				DiseaseName, DiseaseID, AltDiseaseIDs, Def, _, _, _, Synonyms, _ = line.strip('\n').split('\t')
				AllDiseaseIDs = tuple([DiseaseID]+AltDiseaseIDs.split('|')) if AltDiseaseIDs else tuple([DiseaseID])
				AllNames = tuple([DiseaseName]+Synonyms.split('|')) if Synonyms else tuple([DiseaseName])
				entry = MEDIC_ENTRY(DiseaseID,DiseaseName,AllDiseaseIDs,AllNames)
				yield DiseaseID, entry

def concept_obj(dictionary,order=None): # , conf)
    concept_ids = [] # list of all concept ids
    # concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
    concept_names = [] # list of all names, same length as concept_ids
    concept_map = {} # names as keys, ids as concepts

    if order:
        use = order
        logger.info('Re-initializing concept object.')
    else:
        use = dictionary.loaded.keys()

    for k in use:
    # keys not in congruent order! To make them congruent:
    # k,v = zip(*dictionary.loaded.items())
    # k = list(k)
    # k.sort()
        c_id = dictionary.loaded[k].DiseaseID
        # a_ids = dictionary.loaded[k].AllDiseaseIDs
        
        for n in dictionary.loaded[k].AllNames:
            concept_ids.append(c_id)
            # concept_all_ids.append(a_ids)
            concept_names.append(n)
            if n in concept_map: # one name corresponds to multiple concepts
                concept_map[n].append(c_id)
                # logger.warning('{0} already in the dictionary with id {1}'.format(n,concept_map[n]))
            else:
                concept_map[n] = [c_id]

    # save the stuff to object
    from ncbi_normalization import sample
    concept = sample.NewDataSet('concepts')
    concept.ids = concept_ids
    # concept.all_ids = concept_all_ids
    concept.names = concept_names
    concept.map = concept_map
    #concept.tokenize = [nltk.word_tokenize(name) for name in concept_names]
    #concept.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in concept] for concept in concept.tokenize])
    #concept.padded = pad_sequences(concept.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    return concept
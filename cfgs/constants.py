import os

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
if not os.path.exists(PRE_TRAINED_VECTOR_PATH):
    os.makedirs(PRE_TRAINED_VECTOR_PATH)
VECTOR_NAME = 'glove.840B.300d'

DATASET_PATH = 'corpus'
CKPTS_PATH = 'ckpts'
LOG_PATH = 'logs'

DATASET_PATH_MAP = {
    "imdb": os.path.join(DATASET_PATH, 'imdb'),
    "yelp_13": os.path.join(DATASET_PATH, 'yelp_13'),
    "yelp_14": os.path.join(DATASET_PATH, 'yelp_14'),
}

TEXT_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb_text.pt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_13_text.pt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_14_text.pt'),
}
USR_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb_usr.pt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_13_usr.pt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_14_usr.pt'),
}
PRD_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb_prd.pt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_13_prd.pt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp_14_prd.pt'),
}
EXTRA_PRD_VECTORS_MAP = {
    "imdb": os.path.join(PRE_TRAINED_VECTOR_PATH, 'imdb-embedding-200d.txt'),
    "yelp_13": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp-2013-embedding-200d.txt'),
    "yelp_14": os.path.join(PRE_TRAINED_VECTOR_PATH, 'yelp-2014-embedding-200d.txt')
}
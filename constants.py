W2V_FILE = "data/w2v.txt"
GLOVE_FILE = "data/GloVe.txt"
HEADIFIED_FILE = "data/head.txt"
TREES_FILE = "data/trees.txt"
NT_EMB_FILE = "data/xbar.nonterminals"
LEX_FILE = "data/xbar.lexicon"
GR_FILE = "data/xbar.grammar"
TRAIN_FILE = "data/train_small.txt"

PRE_TRAINED_FILE = "output/default_cpu/model_dict"
CORPUS_INFO_FILE = "corpus_info.tar"
MAX_SEN_LENGTH = 50                  # maximum lenght of a sentence
MAX_TEST_SEN_LENGTH = 30
C_UNT = 20
C_P2L = 50
DIMENSION_OF_MODEL = 100
DIM_TERMINAL = 314
MAX_VOCAB_SIZE = 500000

HEADIFY_COMMAND = "perl treebank-scripts-master/headify treebank-scripts-master/newmarked.mrk data/trees.txt > data/head.txt"

UNCOMMON_THRESHOLD = 20
RARE_THRESHOLD = 10
OOV_IDX = 0
OOV = 'OOV'
BOS = 'BOS'
U_TM = 'U_TM'       # unary terminal symbol
U_NTM = 'U_NTM'     # unary nonterminal symbol
TERMINAL = 'TERMINAL'

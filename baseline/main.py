import gr
import time
from ptb import ptb
from util import unbinarize, item_tree, fix_terminals, binarize, oneline
import numpy as np

np.random.seed(21218)

def run(GrammarObject):
    train_inp = ptb("train", 3, 30, 100)
    np.random.shuffle(train_inp)
    train = util.tree2matrix(train_inp)


if __name__=="__main__":
    run(gr.GrammarObject)
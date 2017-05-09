import evalb as ev
from ptb import ptb
import numpy as np
from nltk import Tree
from util import unbinarize, oneline

examples = ptb("test", minlength=3, maxlength=20, n=10)

test = list(examples)
#np.random.shuffle(test)

sample = "(S (INTJ (UH No)) (, ,) (NP (PRP it)) (VP (VBD was) (RB n't)) (NP (NNP Black) (NNP Monday)) (. .))"

'''
for i in xrange(10):
	sen = test[i][0]
	tree = unbinarize(test[i][1])
	print tree, "\n"
'''
sen = test[0][0]
tree = oneline(unbinarize(test[0][1]))

print ev.evalb(tree, Tree.fromstring(sample))

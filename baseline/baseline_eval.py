import gr
import grammar
import time

import evalb
from ptb import ptb
from nltk import Tree
from util import unbinarize, oneline

def run(GrammarObject):
    grammer_obj = GrammarObject()
    t0 = time.time()
    grammer_obj.read_grammar('xbar')
    t1 = time.time()
    #print "Reading grammar takes %.4f secs" % round(t1 - t0, 5)

    begin = time.time()
    examples = ptb("test", minlength=3, maxlength=20, n=10)
    test = list(examples)
    cumul_accuracy = 0
    num_trees_with_parse = 0
    for (sentence, gold_tree) in test:
        log_prob_sentence = grammer_obj.do_inside_outside(sentence)
        #t4 = time.time()
        #print "Inside-outside takes %.4f secs" % round(t4 - t3, 5)

        posterior_threshold = 1e-12
        grammer_obj.prune_the_chart(log_prob_sentence, posterior_threshold)
        #t5 = time.time()
        #print "Pruning takes %.4f secs" % round(t5 - t4, 5)

        str = grammer_obj.parse(sentence)
        #t6 = time.time()
        #print "Parsing takes %.4f secs\n" % round(t6 - t5, 5)

        print "log of Pr( ", "sentence", ") = ", log_prob_sentence
        print str
        
        tree = oneline(unbinarize(gold_tree))
        if str != "":
            tree_accruacy = evalb.evalb(tree, unbinarize(Tree.fromstring(str)))
            cumul_accuracy += tree_accruacy
            num_trees_with_parse += 1
            print tree_accruacy
        else:
            print "No parse!"
        
        # Debug
        # grammer_obj.validate_read_grammar()

    end = time.time()
    print "Parsing takes %.4f secs\n" % round(end - begin, 5)
    print "Fraction of trees with parse = %.4f\n" % round(float(num_trees_with_parse) / len(test), 5)
    accuracy = cumul_accuracy / num_trees_with_parse
    print "Parsing accuracy = %.4f\n" % round(accuracy, 5)

if __name__=="__main__":
    run(gr.GrammarObject)
    
    #run(grammar.GrammarObject)

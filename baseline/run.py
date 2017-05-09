import gr
import grammar
import time

def run(GrammarObject):
    grammer_obj = GrammarObject()
    t0 = time.time()
    grammer_obj.read_grammar('xbar')
    t1 = time.time()
    #print "Reading grammar takes %.4f secs" % round(t1 - t0, 5)

    grammer_obj.compute_sum_and_max_of_unary_combos()
    t2 = time.time()
    #print "Unary combos take %.4f secs" % round(t2 - t1, 5)
    threshold = 1e-7

    grammer_obj.prune_unlikely_rules_and_lexicon(threshold)
    t3 = time.time()
    #print "Pruning rules and lexicon takes %.4f secs" % round(t3 - t2, 5)

    begin = time.time()
    test = "../data/corpus/test.sen"
    with open(test, 'r') as file:
        i = 0
        for sentence in file:
            if i == 5:
                break
            i += 1
            log_prob_sentence = grammer_obj.do_inside_outside(sentence)
            #t4 = time.time()
            #print "Inside-outside takes %.4f secs" % round(t4 - t3, 5)

            posterior_threshold = 0
            grammer_obj.prune_the_chart(sentence, log_prob_sentence, posterior_threshold)
            #t5 = time.time()
            #print "Pruning takes %.4f secs" % round(t5 - t4, 5)

            str = grammer_obj.parse(sentence)
            #t6 = time.time()
            #print "Parsing takes %.4f secs\n" % round(t6 - t5, 5)

            print "log of Pr( ", "sentence", ") = ", log_prob_sentence
            print grammer_obj.debinarize(str)
            
            # Debug
            # grammer_obj.validate_read_grammar()

    end = time.time()
    print "Parsing takes %.4f secs\n" % round(end - begin, 5)
if __name__=="__main__":
    run(gr.GrammarObject)
    
    #run(grammar.GrammarObject)
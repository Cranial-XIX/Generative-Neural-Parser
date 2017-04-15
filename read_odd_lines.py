from __future__ import print_function
import sys

with open(sys.argv[1], 'r') as f:
    line_num = 1
    for line in f:
        if line_num % 2 == 1: # every odd line
            print (line, end='')
        line_num+=1
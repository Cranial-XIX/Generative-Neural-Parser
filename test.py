import numpy as np
from numpy import array
import os
import time
import torch

import constants

def t():
	a = set()
	b = []
	for i in xrange(100000):
		a.add(i)
		b.append(i)
	t1 = time.time()
	c = 0
	for i in a:
		c += i
	t2 = time.time()
	c = 0
	t3 = time.time()
	for i in b:
		c += i
	t4 = time.time()
	print "list operation takes ", (t4 - t3)
	print "set operation takes ", (t2 - t1)
	print "saves ", ((t2-t1)-(t4-t3)) * 40

def test_next(p):
	idx = p.next(0, 2)
	print idx
	print p.p2l.size()
	idx = p.next(idx, 2)
	print idx
	print p.p2l.size()


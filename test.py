import numpy as np
from numpy import array
import os
import time
import torch

import constants

def test_next(p):
	idx = p.next(0, 2)
	print idx
	print p.p2l.size()
	idx = p.next(idx, 2)
	print idx
	print p.p2l.size()


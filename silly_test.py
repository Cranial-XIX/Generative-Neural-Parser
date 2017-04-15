import time
import torch

max = 0
min = 100
for _ in xrange(100):
    a = torch.rand(50, 50)
    b = torch.rand(50, 100)
    c = torch.rand(1, 50)
    d = torch.rand(1, 100)
    t0 = time.time()
    #a.mm(b)
    torch.cat((c,d), 1)
    t1 = time.time()
    timing = t1 - t0
    if timing > max:
        max = timing
    if timing < min:
        min = timing
    print timing
    
print "max, min: ", max, " ", min
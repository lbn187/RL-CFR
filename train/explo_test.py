import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from multiprocessing import Process
import time

def script(test_time):
    os.system("./../src/build/CFR -2 0 1 "+str(test_time))

THREADS = 10
test_time = 100
for epoch in range(10):
    start = time.time()
    print('epoch:',epoch)
    threads = []
    for i in range(THREADS):
        threads.append(Process(target=script, args=(test_time,)))
        threads[i].start()

    for i in range(THREADS):
        threads[i].join()
    end = time.time()
    print(end-start,'s')

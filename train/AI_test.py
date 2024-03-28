import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from multiprocessing import Process
import time

def script(test_time, thread_id):
    os.system("./../src/build/CFR -3 "+str(thread_id)+" 1 0 "+str(test_time))

THREADS = 60
test_time = 1000
for epoch in range(20):
    start = time.time()
    print('epoch:',epoch)
    threads = []
    for i in range(THREADS):
        threads.append(Process(target=script, args=(test_time,i,)))
        threads[i].start()

    for i in range(THREADS):
        threads[i].join()
    end = time.time()
    print(end-start,'s')

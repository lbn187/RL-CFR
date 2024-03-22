import torch
import torchvision
import torch.nn as nn
import numpy as np
from HAND_CRITIC import HAND_CRITIC
import os
from multiprocessing import Process
import time

def script(test_time, thread_id):
    os.system("./../src/CFR_AI "+str(test_time)+" "+str(thread_id))

THREADS = 60
test_time = 1000
for epoch in range(10):
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

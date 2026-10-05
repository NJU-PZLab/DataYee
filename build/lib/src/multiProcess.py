# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from multiprocessing import Pool
import threading, queue
from src.loadjpk import  forcecurve

class multi_run():
    def __init__(self,name):
        super().__init__()
        self.name=name
    def createPool(self,multi_n=4):
        self.multi_n=multi_n
        self.pool = Pool(self.multi_n)
    def inputTask(self,func,arglst):
        res = []
        for arg in arglst:
            res.append(self.pool.apply_async(func,(*arg,)))
        self.pool.close()
        self.pool.join()
        self.results =  [i.get() for i in res]
def splitRange(lens,n):
    if lens==1:
        return [range(0,1)]
    step = int(lens / n) + 1
    lst = []
    for i in range(0,lens,step):
        if i+step<lens:
            lst.append(range(i,i+step))
        else:
            lst.append(range(i,i+step-1))
    return lst
def get_fc(q,ljp,index):
    data = (index,ljp[index])
    q.put(data,timeout=100)
def get_batchfc(q,ljp,index_range):
    if index_range=='all':
        index_range = range(len(ljp))
    q_index = queue.PriorityQueue(maxsize=0)
    [q_index.put(i) for i in index_range]
    while not q_index.empty():
        t = threading.Thread(target=get_fc,args=(q,ljp,q_index.get()))
        t.start()
        t.join()
        
class get_fcdata():
    def __init__(self,ljp):
        self.ljp = ljp
        self.fc = forcecurve
    def create_quene(self,max_size=10):
        self.q = queue.PriorityQueue(maxsize=max_size)
    def put_data(self):
        self.t = threading.Thread(target=get_batchfc,args=(self.q,self.ljp,'all',))
        self.t.start()
    def get_data(self):
        return self.q.get(timeout=5)
if __name__=='__main__':
    pass

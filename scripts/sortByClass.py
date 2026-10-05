# -*- coding: utf-8 -*-
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _sort_key(cls):
    if cls == 'F': return 900
    if cls == 'N': return 1000
    if cls == 'Z': return 1001
    return ord(cls[0]) if cls else 0

class sortByClass():
    def __init__(self, zpo, ljp, fc):
        self.zpo = zpo
        self.lst = []
        zpo.changedforce(save=False)

    def run(self, index):
        d = self.zpo[index]
        self.lst.append(_sort_key(d.get('class', 'N')))

    def end(self):
        pkl = self.zpo.data['data.pkl']
        keys = list(pkl.keys())
        n = len(keys)
        if n != len(self.lst):
            return
        arr = np.array([(self.lst[i], i) for i in range(n)], dtype=[('k', int), ('idx', int)])
        arr.sort(order='k')
        new_pkl = {}
        for row in arr:
            new_pkl[keys[row['idx']]] = pkl[keys[row['idx']]]
        self.zpo.data['data.pkl'] = new_pkl
        self.zpo.savefile()

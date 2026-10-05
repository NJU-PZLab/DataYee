# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:44:37 2021

@author: 13113
"""
import numpy as np
import os
class exportMaxForce():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.lst = []
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata(s=13)['retract']
        x,y  = data['measuredHeight']*1e9,data['vDeflection']*1e12
        i = np.argmax(y[self.fc.data['peakindex']])
        max_force = y[self.fc.data['peakindex'][i]][0]
        k = self.fc.data['k'][i]
        lc = self.fc.data['wlcarg'][i][0]
        if self.fc.data['artificial_judge']:
            j = 1
        else:
            j = 0
        print((index,j,max_force,k,lc))
        self.lst.append((index,j,max_force,k,lc))
        print(index)
    def end(self):
        arr = np.array(self.lst)
        dirname = os.path.dirname(self.zpo.fname)
        fname = os.path.splitext(os.path.basename(self.zpo.fname))[0]+'_MaxForce.txt'
        fname = os.path.join(dirname, fname)
        np.savetxt(fname,arr)
        print(fname)
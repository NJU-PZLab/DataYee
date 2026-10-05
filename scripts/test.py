# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:08:57 2022

@author: zbin
"""
from src.datapro import countdlc,mkbaseondlc
import numpy as np
import copy
class test():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.recover_force()
        d = self.fc.get_prodata()['retract']
        x,y = d['measuredHeight']*1e9,d['vDeflection']*1e12
        p = copy.deepcopy(self.fc.data['peakindex'])
        lens = int(len(y)*0.1)
        y_ = y[p[-1]:p[-1]+lens]
        self.fc.data['offset']['y']-=y_.mean()*1e-12
        while True:
            if x[p[-1]]>150:
                if y[p[-1]]<100:
                    del self.fc.data['peakindex'][-1]
                    del self.fc.data['k'][-1]
                    del self.fc.data['wlcarg'][-1]
                    countdlc(self.fc)
                    mkbaseondlc(self.fc)
                    del p[-1]
                else:
                    break
            else:
                break
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce(save=False)
        
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:18:38 2022

@author: zbin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:08:57 2022

@author: zbin
"""
from src.datapro import countdlc,mkbaseondlc,Lc_transformer_
import numpy as np
import copy
class test1():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.recover_force()
        self.fc.data['class'] = 'F'
        d = self.fc.get_prodata()['retract']
        x,y = d['measuredHeight']*1e9,d['vDeflection']*1e12
        y[np.where(x>250)[0]] = 0
        lc = Lc_transformer_(x,y)
        lc = lc[np.where(lc>30)[0]]
        if len(lc)>1:
            dlc = np.diff(lc)
            if len(dlc)<=8 and len(np.where((dlc>15)&(dlc<24))[0])>=2:
                self.fc.data['class']='D'
        self.zpo.changingforce(self.fc)
        print(dlc)
        print(index)
    def end(self):
        self.zpo.changedforce(save=False)
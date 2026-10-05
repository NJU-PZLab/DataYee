# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 10:41:38 2022

@author: zbin
"""

class reClassify_H3_240_282_unfolding():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.lst = []
    def run(self,index):
        self.fc.data = self.zpo[index]
        mark = self.fc.data['mark']
        #self.fc.recover_force()
        #data = self.fc.get_prodata()['retract']
        #x,y = data['measuredHeight']*1e9,data['vDeflection']*1e12
        #peakf = y[self.fc.data['peakindex']]
        n_GB1,n_h3_0 = mark.count('GB1'),mark.count('h3_0')
        print(n_GB1)
        if self.fc.data['class'] =='F':
            self.fc.data['class'] = 'F'
        elif 'h3_0' in mark:
            if n_GB1 == 3:
                self.fc.data['class'] = 'A'
            elif n_GB1 == 2:
                self.fc.data['class'] = 'B'
        elif 'none' not in mark:
            if n_GB1 == 3:
                self.fc.data['class'] = 'C'
            elif n_GB1 == 2:
                self.fc.data['class'] = 'D'
                
        self.fc.clean_force()    
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce(save=False)
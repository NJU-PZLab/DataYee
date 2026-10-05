# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 10:41:38 2022

@author: zbin
"""

class reClassify_H3_unfolding():
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
        elif 'none' in mark:
            self.fc.data['class'] = 'N'
        elif 'H3_0' in mark and "H3" not in mark:
            self.fc.data['class'] = 'C'
        elif 'H3_0' in mark and "H3" in mark:
            self.fc.data['class'] = 'B'
        elif 'H3_0' not in mark and "H3" in mark:
            self.fc.data['class'] = 'A'
        else:
            self.fc.data['class'] = 'D'
                
        self.fc.clean_force()    
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce(save=False)
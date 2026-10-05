# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 10:41:38 2022

@author: zbin
"""

class reClassify_H3():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.lst = []
    def run(self,index):
        self.fc.data = self.zpo[index]
        mark = self.fc.data['mark']
        self.fc.recover_force()
        data = self.fc.get_prodata()['retract']
        x,y = data['measuredHeight']*1e9,data['vDeflection']*1e12
        peakf = y[self.fc.data['peakindex']]
        if self.fc.data['class'] != 'F':
            self.fc.data['class'] = 'N'
        if self.fc.data['class'] in ['F']:
            pass
        elif max(peakf)>280:
            self.fc.data['class'] = 'E'
        else:
            if len(mark) !=0:
                dlc = self.fc.data['dlc']
                if len(mark)==1:
                    if dlc>10 and dlc<24:
                        self.fc.data['class'] = 'C'
                    else:
                        self.fc.data['class'] = 'D'
                else:
                    self.fc.data['class'] = 'I'
                    
            else:
                lc = self.fc.data['wlcarg'][-1][0]
                if lc<25:
                    self.fc.data['class'] = 'G'
                elif lc>25.01 and lc<46:
                    self.fc.data['class'] = 'A'
                elif lc>46.01 and lc <70:
                    self.fc.data['class'] = 'B'
        self.fc.clean_force()    
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce(save=False)
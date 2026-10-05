# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 10:41:38 2022

@author: zbin
"""

class reClassify():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.lst = []
    def run(self,index):
        self.fc.data = self.zpo[index]
        mark = self.fc.data['mark']
        if self.fc.data['class'] in ['F']:
            pass
        else:
            if 'RBD' in mark:
                self.fc.data['class'] = 'A'
            elif 'RBD-1' in mark:
                self.fc.data['class'] = 'C'
            elif 'RBD-0' in mark:
                self.fc.data['class'] = 'E'
            elif 'none' not in mark and 'GB1' in mark:
                self.fc.data['class'] = 'D'
            else:
                self.fc.data['class'] = 'N'
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce(save=False)
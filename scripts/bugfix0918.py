# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:39:32 2021

@author: zbin
"""


class bugfix0918():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.clean_force()
        self.zpo.changingforce(self.fc)
    def end(self):
        self.zpo.changedforce()
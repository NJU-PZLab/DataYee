# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:20:06 2021

@author: 13113
"""

"""
Created on Sat Sep 11 16:08:18 2021

@author: 13113
"""
outputname = 'test123.txt'
import sys
sys.path.append("../src")
from src.datapro import wlcfit,findpeak,slope,countdlc,mkbaseondlc,peakH,cal_baseline_y
#force,k,correct
import numpy as np
class changeOriginPoint2startpoint():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_x,data_y = data['measuredHeight'],data['vDeflection']
        x = data_x[np.argmin(data_y)]
        self.fc.data['arg']['mark'] = {'GB1': (15.0, 22.99),'RBD-0': (6.0, 14.99),'RBD-1': (23.0, 50),'RBD': (50.01, 80)}
        cal_baseline_y(self.fc)
        findpeak(self.fc)
        peakH(self.fc)
        wlcfit(self.fc)
        slope(self.fc)
        countdlc(self.fc)
        mkbaseondlc(self.fc)
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce()
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:08:18 2021

@author: 13113
"""
outputname = 'test123.txt'
#force,k,correct
import numpy as np
class cal_highspeed_correct():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.arr = np.array([])
    def run(self,index):
        self.fc.data = self.zpo[index]
        if not self.fc.data['artificial_judge']:
            return None
        try:
            peakindex = self.fc.data['peakindex']
            self.fc.recover_force(self.ljp)
            y_retrace = self.fc.get_prodata()['retract']['vDeflection']
            force = y_retrace[peakindex[-1]][0]
            y_trace = self.fc.get_prodata()['extend']['vDeflection']
            len_retrace = int(len(y_retrace)*0.9)
            len_trace = int(len(y_trace)*0.1)
            corr = y_retrace[len_retrace:].mean()-y_trace[:len_trace].mean()
            k = self.fc.data['k'][-1]
            self.fc.data['offset']['highspeed'] = 0.5*corr
            d = np.array([index,force,k,corr])
            print(d)
            if len(self.arr)==0:
                self.arr = d
            else:
                self.arr = np.vstack((self.arr,d))
        except Exception as err:
            print(err)
    def end(self):
        np.savetxt(outputname,self.arr,fmt='%.3e')
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:44:37 2021

@author: 13113
"""
import numpy as np
def get_slope(x_arr,y_arr,index=-1):
    p = np.polyfit(x_arr,y_arr,1)
    d = np.polyder(p)
    k = np.polyval(d, x_arr[index])
    return k
class align_cellcurve():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.arr = np.array([])
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.recover_force(self.ljp)
        p = self.fc.data['peakindex']
        if len(p)>0:
            self.fc.data['offset']['k'] = 0
            data = self.fc.get_prodata()['retract']
            data_y,data_x = data['vDeflection'],data['measuredHeight']
            lens = len(data_y)
            range_len = int(0.2*lens)
            i = p[-1]
            if i+range_len<lens:
                x,y = data_x[i:i+range_len],data_y[i:i+range_len]
                print(i,i+range_len)
            else:
                x,y = data_x[i:],data_y[i:]
            k = get_slope(x.reshape(-1), y.reshape(-1))
            self.fc.data['offset']['k'] = k
            data = self.fc.get_prodata()['retract']
            data_y,data_x = data['vDeflection'],data['measuredHeight']
            if i+range_len<lens:
                x,y = data_x[i:i+range_len],data_y[i:i+range_len]
            else:
                x,y = data_x[i:],data_y[i:]
            self.fc.data['offset']['y']-=np.mean(y)
        else:
            pass
        self.fc.clean_force()
        self.zpo.changingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce()
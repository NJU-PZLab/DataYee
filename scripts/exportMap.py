# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 09:14:29 2021

@author: zbin
"""

import matplotlib.pyplot as plt
import numpy as np
width = 32
class exportMap():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.d = np.zeros(len(self.ljp))
    def run(self,index):
        print(index)
        self.fc.data = self.zpo[index]
        i = self.fc.data['datamsg'][1]
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_y = data['vDeflection']*1e12
        self.d[i] = data_y.max()
        self.fc.clean_force()
    def end(self):
        fig,ax = plt.subplots(figsize=(4,4),dpi=300)
        arr = np.zeros((len(self.d)//width,width))
        self.d = self.d[:len(self.d)//width*width]
        for i,data in enumerate(self.d):
            if (i//width)%2==0:
                arr[i//width,i-(i//width)*width]=data
            else:
                arr[i//32,31-(i-(i//32)*32)]=data
        ax.imshow(arr)
        fig.savefig('map.png',bbox_inches='tight',transparent=True)
        pass
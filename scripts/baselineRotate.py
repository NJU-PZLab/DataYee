# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:36:44 2022

@author: DELL
"""
import numpy as np
from scipy.optimize import curve_fit
def get_slope(x_arr,y_arr,index=-1):
    p = np.polyfit(x_arr,y_arr,1)
    d = np.polyder(p)
    k = np.polyval(d, x_arr[index])
    return k
def lcfunc(x, lc, lp):
    return 1.3806e-23 * 298 / (lp * 1e-9) * (1 / 4 * (1 - x / lc) ** (-2) + x / lc - 1 / 4) * 1e12

def WRC_transformer(f,x,thr=20):
    b,gama = 0.11e-9,41/180*np.pi
    kb = 1.38e-23
    T = 298
    x = x[np.where(f>thr)]*1e-9
    f = f[np.where(f>thr)]*1e-12
    l=b*np.cos(gama/2)/np.abs(np.log(np.cos(gama)))
    f_b = kb*T*l/b**2
    x1 = x[np.where(f<f_b)]/(1-(4*f[np.where(f<f_b)]*l/kb/T)**(-0.5))
    x2 = x[np.where(f>=f_b)]/(1-(2*f[np.where(f>=f_b)]*b/kb/T)**(-1))
    return np.hstack((f[np.where(f<f_b)],f[np.where(f>=f_b)]))*1e12,np.hstack((x1,x2))*1e9
def wlcfit(fc):
    fc.data['wlcarg'] = []
    fc.data['slopepre'] = []
    n = 50
    data = fc.get_prodata()['retract']
    data_x,data_y = data['measuredHeight'].reshape(-1)*1e9,data['vDeflection'].reshape(-1)*1e12
    lp=fc.data['arg']['lp']
    for i,p_i in enumerate(fc.data['peakindex']):
        try:
            b_i = np.where(p_i>fc.data['bottomindex'])[0]
        except:
            b_i = []
        if len(b_i)!=0:
            b_i = fc.data['bottomindex'][b_i[-1]]
        else:
            temp = p_i
            while temp-n>0:
                k = get_slope(data_x[temp - n:temp], data_y[temp - n:temp])
                if k < 0.01 or data_x[temp] < 10:
                    if temp == p_i:
                        temp -= 10
                    break
                else:
                    temp -= n
            b_i = temp
            if b_i==p_i:
                b_i-=3
        if data_y[p_i]>150:
            dy = data_y[p_i]-data_y[b_i]
            fitpoint = np.where(data_y[b_i:]>data_y[b_i]+0.6*dy)[0][0]+b_i
        else:
            fitpoint = p_i
        if fitpoint==b_i:
            fitpoint+=20
        try:
            popt, _ = curve_fit(lcfunc, data_x[b_i:fitpoint], data_y[b_i:fitpoint],
                                    bounds=([data_x[p_i], lp[0]], [data_x[p_i] + 50, lp[1]]))
        except:
            popt = (WRC_transformer(data_y[b_i:fitpoint],data_x[b_i:fitpoint],thr=5)[1].mean(),0.36)
        try:
            popt_pre, _ = curve_fit(lcfunc, data_x[b_i:p_i], data_y[b_i:p_i],
                                    bounds=([data_x[p_i], 0], [data_x[p_i] + 50, 0.5]))
        except:
            popt_pre = (WRC_transformer(data_y[b_i:p_i],data_x[b_i:p_i],thr=5)[1].mean(),0.36)
        lc, p = popt
        fc.data['wlcarg'].append((lc, p))
        fc.data['slopepre'].append((popt_pre[0],popt_pre[1]))
    fc.data['wlcarg'] = list(fc.data['wlcarg'])
class baselineRotate():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        data = self.zpo[index]
        self.fc.data = data
        if len(self.fc.data['peakindex'])==0:
            self.zpo.deletingforce(self.fc)
            return None
        self.fc.recover_force(self.ljp)
        s,e = self.fc.data['peakindex'][-1]+20,-1
        self.fc.data['offset']['k'] = 0
        d = self.fc.get_prodata()['retract']
        x,y = d['measuredHeight'],d['vDeflection']
        k = get_slope(x[s:e].reshape(-1),y[s:e].reshape(-1),0)
        self.fc.data['offset']['rotate_index'] = s
        self.fc.data['offset']['k']=k
        d = self.fc.get_prodata()['retract']
        x,y = d['measuredHeight'],d['vDeflection']
        self.fc.data['offset']['y'] -= y[s:e].mean()
        xx = x[0]
        self.fc.data['offset']['x'] = self.fc.data['offset']['x']+xx
        self.fc.data['arg']['lp'] = [0.3,0.42]
        wlcfit(self.fc)
        d = self.fc.get_prodata()['retract']
        x,y = d['measuredHeight'],d['vDeflection']
        max_i = np.argmax(y[self.fc.data['peakindex']])
        lc = self.fc.data['wlcarg'][max_i][0]
        if lc >=35 and lc <=100 and y[self.fc.data['peakindex'][max_i]]*1e12>30:
            self.zpo.changingforce(self.fc)
        else:
            self.zpo.deletingforce(self.fc)
        print(index)
    def end(self):
        self.zpo.changedforce()
        self.zpo.deletedforce()
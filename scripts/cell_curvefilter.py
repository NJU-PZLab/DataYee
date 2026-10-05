# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:24:07 2021

@author: zbin
"""

import torchvision.transforms as transforms
from PIL import Image
import torch
from PIL import ImageOps
import matplotlib.pyplot as plt
from scipy import signal
from tslearn.preprocessing import TimeSeriesResampler
import numpy as np
from scipy.signal import  find_peaks
from scipy.ndimage import gaussian_filter
def mean(lst):
    return sum(lst)/len(lst)
plt.ioff()
def fig2img(fig):
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return img
def couple(lst):
    r = []
    lst.insert(0,0)
    lst.append(-1)
    for i,v in enumerate(lst):
        if i+1<len(lst):
            r.append((v,lst[i+1]))
    return r
def diff(arr,n):
    if n%2==0:
        return False
    lst = list(arr)
    pend = int((n-1)/2)
    arr = np.array([mean(lst[0:pend])]*pend+lst+[mean(lst[-pend:])]*pend)
    arr_new = np.array([])
    for i in range(pend,len(arr)-pend):
        data = arr[i-pend:i+pend+1]
        p = np.polyfit(np.arange(len(data)),data,1)
        d = np.polyder(p)
        k = np.polyval(d,arr[i])
        arr_new = np.append(arr_new,k)
    return arr_new
def cell_predict(s1,s2):
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(r'./script/2021-08-29-16-1.7.1+cpuacc84.model', map_location='cpu', weights_only=False)
    fig,ax = plt.subplots(figsize=(6,6),dpi=300)
    plt.subplots_adjust(left=0, bottom=0, right=1,top=1,hspace=0.1,wspace=0.1)
    ax.plot(s1,s2,lw=2.5,color='k')
    img = fig2img(fig).resize((224,224)).convert('RGB')
    img = ImageOps.invert(img)
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    return preds
def min_odd(n):
    if n%2==0:
        n+=1
    return n
def cell_curve(fc,debug=False):
    cut_num = 300
    b, a = signal.butter(8, 0.1, 'lowpass')
    data = fc.get_prodata(s=19)['retract']
    data_x,data_y = data['measuredHeight']*1e9,data['vDeflection']*1e12
    data_y_cut = TimeSeriesResampler(1000).fit_transform(data_y[cut_num:].reshape(-1)).reshape(-1)
    d2 = signal.filtfilt(b, a, diff(diff(data_y_cut,51),5))*-100
    p,_ = find_peaks(d2,height=1.5,width=15)
    if debug:
        plt.plot(d2)
        plt.plot(np.arange(len(d2))[p],d2[p],'ro')
        plt.show()
    '''if len(p)>4:
        p = np.sort(p[np.argsort(_['prominences'])[-5:]])'''
    p_new = np.array([])
    for i in p:
        if i+40 < len(d2)-1 and d2[i+np.argmin(d2[i:i+40])]<-1.5:
            p_new = np.append(p_new,i)
        else:
            for n in range(i,len(d2)):
                if n+1<len(d2) and d2[n]-d2[n+1]>0:
                    continue
                else:
                    if d2[n] < -1:
                        p_new = np.append(p_new,i)
                    break
    ranges =  couple(list(p_new.astype(np.int32)))
    if debug:
        print(ranges)
    result_lst = []
    for i,_ in enumerate(ranges):
        if i+1<len(ranges):
            s1 = data_y_cut[ranges[i][0]:ranges[i][1]]
            s2 = data_y_cut[ranges[i+1][0]:ranges[i+1][1]]
            s = np.append(s1,s2)
            d = diff(s,13)*-1
            d[np.where(d<0)]=0
            if d.max()<1.5*np.percentile(d,90):
                continue
            result_lst.append((ranges[i][0],ranges[i][1],ranges[i+1][1]))
    result_arr = (np.array(result_lst)*len(data_y)/1000).astype(np.int32)+cut_num
    for i,v in enumerate(result_arr):
        start,middle,end = v
        if end < middle:
            end = len(data_y)-1
            v[2] = len(data_y)-1
        d = np.diff(gaussian_filter(data_y[start:end].reshape(-1).reshape(-1),min_odd(int(len(data_y)/150))),2)
        if debug:
            plt.plot(d)
            plt.show()
        p,_ = find_peaks(d,width=15)
        p = np.sort(p[np.argsort(_['prominences'])[-3:]])
        dm =  p[np.argsort(abs(p+start-middle))]+start-middle
        v = v + dm[np.argmin(abs(dm))]
        result_arr[i] = v
    if debug:
        print(result_arr)
    return result_arr
class cell_curvefilter():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        self.fc.data['filters']['win_lens'] = 21
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata(s=25)['retract']
        data_x,data_y = data['measuredHeight']*1e9,data['vDeflection']*1e12
        try:
            r = cell_curve(self.fc)
            if len(r)<10:
                predict_result = [cell_predict(data_x[v[0]:v[2]],data_y[v[0]:v[2]]) for v in r]
            else:
                self.zpo.deletingforce(self.fc)
            if not  sum(predict_result)!=len(predict_result):
                self.zpo.deletingforce(self.fc)
        except:
            self.zpo.deletingforce(self.fc)
        self.fc.clean_force()
        print("{}/{}".format(index, len(self.zpo)))
    def end(self):
        self.zpo.deletedforce()
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:09:11 2021

@author: ZhengBin
"""
from sklearn.cluster import KMeans
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from sklearn.neighbors import KernelDensity
import numpy as np
from src.loadjpk import forcecurve,loadjpkfile,zipfileopera
#from dtaidistance import dtw
from src.datapro import wlc2lc,cal_baseline_x,cal_baseline_y

def WLC_transformer(f,x,thre=30):
    x = x[np.where(f>thre)].astype(complex)*1e-9
    f = f[np.where(f>thre)].astype(complex)*1e-12
    p = np.array([0.36e-9],dtype=complex)
    lc = wlc2lc(x,f,p)
    return f,lc
def Lc_transformer(data_x,data_y,length=400,step=2,thre=30):
    f,x = WLC_transformer(data_y,data_x,thre=thre)
    x,f = x.real*1e9,f.real*1e12
    if len(x)<=10:
        return np.zeros(len(np.arange(0,length,step)))
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(x.reshape(-1,1))
    x_ = np.arange(0,length,2)
    log_dens = kde.score_samples(x_.reshape(-1,1))
    return  np.exp(log_dens)
def count_0(x):
    x_ = np.array([])
    for i in x:
        if len(x_)==0:
            x_ = i.reshape(1,2)
        if i[0]>x_[:,0].max() and i[1]>x_[:,1].max():
            x_ = np.vstack((x_,i))
    return len(x_)

def wlc_dist(s1,s2,dlc_thre=5,f_thre=30):
    s1,s2 = np.delete(s1,np.where(s1==0)[0]),np.delete(s2,np.where(s2==0)[0])
    s1_dlc,s2_dlc = s1[:len(s1)//2],s2[:len(s1)//2]
    score = max(len(s1_dlc),len(s2_dlc))
    s1_ = np.tile(s1_dlc,(len(s2_dlc),1))
    s2_ = np.tile(s2_dlc.reshape(-1,1),(1,len(s1_dlc)))
    matrix_dlc = np.abs(s1_-s2_)
    arr_coor = np.dstack(np.where(matrix_dlc<=dlc_thre))[0]
    reduct = max(count_0(arr_coor),count_0(arr_coor[arr_coor[:,1].argsort()]))
    return 1-reduct/score

def distance(s1,s2):
    d = dtw.distance(s1,s2,window=int(0.25*len(s1)), penalty=0.2,use_c=True)
    return d
def get_lcseq(zpo,ljp,indexlst,length,step,thre):
    fc = forcecurve()
    arr = np.array([])
    for i,index in enumerate(indexlst):
        fc.data = zpo[index]
        fc.recover_force(ljp)
        data = fc.get_prodata()['retract']
        data_x,data_y = data['measuredHeight']*1e9,data['vDeflection']*1e12
        res = Lc_transformer(data_x,data_y,length=length,step=step,thre=thre)
        if res.max()>0:
            res = res/res.max()
        if len(arr)!=0:
            try:
                arr = np.vstack((arr,res))
            except Exception as err:
                print(i,err)
        else:
            arr = res
        
    #arr = np.array(lst)
    return arr
def get_distmatrix(zpo,ljp,m_run,length=400,step=2,thre=30,parallel=True,multip_n=1):
    if not parallel:
        arr = np.array(get_lcseq(zpo,ljp,range(len(zpo)),length,step,thre))
    else:
        print('multiprocessing')
        lens = len(zpo)
        step = int(lens / multip_n) + 1
        arg_lst = []
        m_run.createPool(multip_n)
        for i in range(0,lens,step):
            if i+step<len(zpo):
                arg_lst.append((zpo,ljp,range(i,i+step),length,step,thre))
            else:
                arg_lst.append((zpo,ljp,range(i,len(zpo)),length,step,thre))
        m_run.inputTask(get_lcseq,arg_lst)
        arr = np.vstack(m_run.results)
    matrix = dtw.distance_matrix(arr,window=25,penalty=0.2,use_c=True,parallel=True)
    return matrix
def sort_similar(index,matrix):
    arr = matrix[index,:]
    return arr.argsort()
def KMsClustering(matrix,n_clusters=8):
    km = KMeans(n_clusters=n_clusters,precompute_distances=True).fit(matrix)
    return km.labels_
def multi_run(zpo,ljp,length,step,thre,multip_n=4):
    from multiprocessing import Pool
    lens = len(zpo)
    step = int(lens / multip_n) + 1
    lst = []
    for i in range(0,lens,step):
        if i+step<len(zpo):
            lst.append(range(i,i+step))
        else:
            lst.append(range(i,i+step-1))
    pool = Pool(len(lst))
    result = []
    print(lst)
    for indexlst in lst:
        result.append(pool.apply_async(get_lcseq,(zpo,ljp,indexlst,400,2,30,)))
    pool.close()
    pool.join()
    #arr = np.array([r.get() for r in result])
    return result
    
        
if __name__=='__main__':
    pass
r'''
    zpo = zipfileopera('D:/code/py/DataYee/test.DataYee-force')
    ljp = loadjpkfile(zpo.get_sourcepath())
    import time
    t1=time.time()
    res = multi_run(zpo, ljp, 400, 2, 30)
    arr = np.vstack([r.get() for r in res])
    #res = get_distmatrix(zpo,ljp)
    t2=time.time()
    print(t2-t1)'''
        
        
        
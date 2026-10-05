# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:05:22 2021

@author: Administrator
"""
from loadjpk import forcecurve,loadjpkfile,zipfileopera
from multiProcess import multi_run,splitRange
import numpy as np
import time
def MergeDic(dic_lst):
    for i,dic in enumerate(dic_lst):
        if i==0:
            res_dic = dic
        else:
            res_dic.update(dic)
    return res_dic
        
def get_data(zpo,ljp,outrange,tasktype='smfs'):
    fc = forcecurve()
    data_dic = {}
    for i in outrange:
        fc.data = zpo[i]
        if not fc.data['artificial_judge']:
            continue
        fc.recover_force(ljp)
        data = fc.get_prodata()['retract']
        data_y = data['vDeflection']*1e12
        peakindex = fc.data['peakindex']
        f = (data_y[peakindex]+fc.data['offset']['highspeed']).reshape(-1)
        k = fc.data['k']
        data_dic[i] = dict(force=f,k=k,peakN=len(peakindex))
        if tasktype == 'smfs':
            lc,lp = np.array(fc.data['wlcarg'])[:,0],np.array(fc.data['wlcarg'])[:,1]
            dlc = fc.data['dlc']
            mark = fc.data['mark']
            data_dic[i]['lc'] = lc
            data_dic[i]['lp'] = lp
            data_dic[i]['dlc'] = dlc
            data_dic[i]['mark'] = mark
    return data_dic
def multi_getdata(zpo,ljp,tasktype='smfs',multi_n=4):
    range_lst = splitRange(len(zpo),multi_n)
    arg_lst = list(zip([zpo]*len(range_lst),[ljp]*len(range_lst),range_lst))
    m_r = multi_run('None')
    m_r.createPool(multi_n)
    m_r.inputTask(get_data,arg_lst)
    return m_r.results
if __name__ == '__main__':
    zpo = zipfileopera(r'E:\ZB\DataYee/test.DataYee-force')
    ljp = loadjpkfile(zpo.get_sourcepath())
    index = len(zpo)-1
    t1 = time.time()
    data = MergeDic(multi_getdata(zpo,ljp,tasktype='smfs',multi_n=4))
    #data = get_data(zpo,ljp,range(len(zpo)),tasktype='smfs')
    t2 = time.time()
    print(t2-t1)
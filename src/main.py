import time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import copy
import os
from src.loadjpk import forcecurve,loadjpkfile,zipfileopera
from src.datapro import noise_down,cal_baseline_drift,cal_baseline_x,cal_baseline_y,\
    cal_highspeed_drift,predict,findpeak,wlcfit,peakH,peakN,slope,countdlc,mkbaseondlc
from src.datapro import Lc_transformer,plotmap,plothist,findpeak_smallrange,get_slope,wlc2lc
from src.datapro import extract_curve_features,train_classifier,predict_curve_class
from src.datapro import learn_class_stats,rescue_reanalyze
from src.clusterscore import get_distmatrix,sort_similar,KMsClustering
func_lst = [noise_down,cal_baseline_drift,cal_baseline_y,cal_baseline_x,\
    cal_highspeed_drift,predict,findpeak,peakH,wlcfit,peakN,slope,countdlc,mkbaseondlc]
def process_customize(fc,functions=[0]):
    for i in functions:
        func_lst[i](fc)

def _sort_key(cls):
    if cls == 'F':
        return 900
    if cls == 'N':
        return 1000
    if cls == 'Z':
        return 1001
    return ord(cls[0]) if cls else 0

# ---- Multi-process worker ----
_ljp = None

def _worker_init(source_path):
    global _ljp
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _ljp = loadjpkfile(source_path)

def _worker_process(args):
    index, taskarg_dict, tasktype = args
    global _ljp
    fc = forcecurve()
    try:
        fc.data = _ljp[index]
        fc._prodata_cache.clear()
        fc._prodata_dirty = True
        taskarg_dict['usemodel'] = False
        fc.data['arg'] = taskarg_dict
        fc.data['tasktype'] = tasktype
        if tasktype == 'smfs':
            if len(fc.data.get('rawdata', {})) == 0:
                return (index, None, 0.0)
            try:
                process_customize(fc, [0, 2, 3, 4, 6, 7, 9])
            except Exception:
                return (index, None, 0.0)
            if not fc.data.get('peaknum_judge', False):
                return (index, None, 0.0)
            try:
                process_customize(fc, [8, 10, 11, 12])
            except Exception:
                return (index, None, 0.0)
            if taskarg_dict.get('filter_dlc', False):
                marks = fc.data.get('mark', [])
                if not marks or all(m == 'none' for m in marks):
                    fc.data['class'] = 'Z'
        elif tasktype == 'cell_curve':
            if fc.data.get('rawdata', {}).get('retract', {}).get('vDeflection', np.array([0])).sum() == 0:
                return (index, None, 0.0)
            try:
                process_customize(fc, [0, 1, 2, 3, 4, 6, 9, 10])
            except Exception:
                return (index, None, 0.0)
            if not fc.data.get('peaknum_judge', False):
                return (index, None, 0.0)
        fc.compress()
        fc.clean_force()
        hs = fc.data['offset'].get('highspeed', 0) if taskarg_dict.get('highspeed', False) else 0
        return (index, copy.deepcopy(fc.data), float(hs))
    except Exception:
        return (index, None, 0.0)

def main_smfs(fc,zpo):
    if fc.data['rawdata'] == {}:
        return None
    fc.data['tasktype']='smfs'
    try:
        #0:noise_down,2:cal_baseline_y,3:cal_baseline_x,4:cal_highspeed_drift
        #6:findpeak,7:peakH,9:peakN
        process_customize(fc,[0,2,3,4,6,7,9])
    except Exception as err:
        print(err)
        return None
    if not fc.data['peaknum_judge']:
        return None
    if fc.data['arg']['usemodel']:
        process_customize(fc, [5])
    if not fc.data['mobilenet_judge']:
        return None
    try:
        #8:wlc,10:slope,11:countdlc,12:mkbaseondlc
        process_customize(fc,[8,10,11,12])
    except Exception as err:
        print(err)
        return None
    if fc.data['arg'].get('filter_dlc', False):
        marks = fc.data['mark']
        if not marks or all(m == 'none' for m in marks):
            fc.data['class'] = 'Z'
    zpo.changingforce(fc)
def main_cell(fc,zpo):
    if fc.data['rawdata'] == {} or fc.data['rawdata']['retract']['vDeflection'].sum()==0:
        return None
    fc.data['tasktype']='cell_curve'
    try:
        process_customize(fc, [0,1,2,3,4,6,9,10])
    except:
        return None
    if not fc.data['peaknum_judge']:
        return None
    zpo.changingforce(fc)
def main(fc,zpo,tasktype='smfs'):
    if tasktype=='smfs':
        main_smfs(fc,zpo)
    elif tasktype=='cell_curve':
        main_cell(fc,zpo)
class programbody():
    def __init__(self):
        self.tasktype = 'smfs'
        self.forcecurve_index = 0
        self.forcepeak_index = 0
        self.Realpeakindex = 0
        self.ready_run = False
        self.state = False
        self.fixlc_changelp = False
        self.coor_data = (1,1)
        self.change_dic = {}
        self.dirty = False
        self._undo_stacks = {}
        self._max_undo = 50
        self.highspeedcorr = np.array([])
        self.taskarg = {'peakH': 30,
           'sens': 10,
           'peakN': [1, 6],
           'xlim': 20,
           'lp': [0.34, 0.38],
           'mark': {'GB1': (13, 23), 'I27': (23, 36)},
           'fitjudge': False,
           'usemodel':True,
           'xsens':2,
           'highspeed':False,
           'modelstrict':False,
            'fastmode':'img',
             'filter_dlc': True,
             'sg_win_lens': 0,
              'include_last': True}
    def creattask(self,path,tasktype='smfs'):
        self.tasktype = tasktype
        self.fc = forcecurve()
        if path.endswith('.DataYee-force'):
            self.zpo = zipfileopera(path)
            if len(self.zpo)==0:
                return None
            self.ljp = loadjpkfile(self.zpo.get_sourcepath())
            if 'tasktype' in self.zpo[0].keys():
                self.tasktype = self.zpo[0]['tasktype']
            self.forcecurve_index=0
            self.forcepeak_index=0
            self.state = True
            self.change_dic = {}
        else:
            self.zpo= zipfileopera()
            self.ljp = loadjpkfile(path)
            self.forcecurve_index=0
            self.forcepeak_index=0
            self.ready_run = True
    def curve_change(self):
        if not self.state:
            return None
        self.change_dic[self.forcecurve_index]=self.fc.data['datamsg']
        self.fc.clean_force()
        self.zpo.changingforce(self.fc)
        self.dirty = True

    def _push_undo(self):
        if not self.state:
            return
        key = self.fc.data.get('datamsg', ('', 0))
        if key == ('', 0):
            return
        if key not in self._undo_stacks:
            self._undo_stacks[key] = []
        snapshot = {
            'data': copy.deepcopy(self.fc.data),
            'forcecurve_index': self.forcecurve_index,
            'forcepeak_index': self.forcepeak_index,
        }
        self._undo_stacks[key].append(snapshot)
        if len(self._undo_stacks[key]) > self._max_undo:
            self._undo_stacks[key].pop(0)

    def undo(self):
        if not self.state:
            return False
        key = self.fc.data.get('datamsg', ('', 0))
        if key == ('', 0):
            return False
        if key not in self._undo_stacks or len(self._undo_stacks[key]) == 0:
            return False
        snapshot = self._undo_stacks[key].pop()
        self.fc.data = copy.deepcopy(snapshot['data'])
        self.forcecurve_index = snapshot['forcecurve_index']
        self.forcepeak_index = snapshot['forcepeak_index']
        self.curve_change()
        return True
    def fc_indexchange(self,n=0):
        if not self.state:
            return None
        if self.forcecurve_index+n>len(self.zpo)-1:
            self.forcecurve_index = len(self.zpo)-1
        elif self.forcecurve_index+n<0:
            self.forcecurve_index = 0
        else:
            self.forcecurve_index = self.forcecurve_index+n
        self.forcepeak_index = 0
    def pk_indexchange(self,n,coor=(None,None)):
        if not self.state:
            return None
        peaklength = len(self.fc.data['peakindex'])
        if peaklength == 0:
            return None
        if n==None and coor[0]!=None and coor[1]!=None:
            datax,datay=coor
            self.fc.recover_force(self.ljp)
            data = self.fc.get_prodata()['retract']
            data_x,data_y = data['measuredHeight'].reshape(-1)*1e9,data['vDeflection'].reshape(-1)*1e12
            index = np.argmin(np.abs(data_x[np.argmin(np.abs(datax-data_x))]-data_x[self.fc.data['peakindex']]))
            self.forcepeak_index=index
            self.fc.clean_force()
            return True
        if self.forcepeak_index+n >peaklength-1:
            self.forcepeak_index = peaklength-1
        elif self.forcepeak_index+n<0:
            self.forcepeak_index=0
        else:
            self.forcepeak_index=self.forcepeak_index+n
    def baseline_change(self,n):
        if not self.state:
            return None
        self._push_undo()
        self.fc.data['offset']['y'] += n
        if self.tasktype == 'smfs':
            if 'retract' not in self.fc.data['rawdata'].keys():
                self.fc.recover_force(self.ljp)
            self.fc.data['arg'] = self.taskarg
            cal_baseline_x(self.fc)
            process_customize(self.fc,range(6,13))
        self.curve_change()
    def rebaseline_cal(self,datax1,datax2,allowRotate=False):
        if not self.state:
            return None
        self._push_undo()
        if datax1==datax2:
            return None
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_x,data_y = data['measuredHeight']*1e9,data['vDeflection']*1e12
        if datax2>datax1:
            datax2,datax1=datax1,datax2
        i_start = np.where(data_x>datax2)[0]
        i_end = np.where(data_x<datax1)[0]
        if len(i_start)==0 or len(i_end)==0:
            return None
        else:
            i_start = i_start[0]
            i_end = i_end[-1]
        if i_end-i_start<5:
            return None
        #data_y = self.fc.data['rawdata']['retract']['vDeflection']
        self.fc.data['offset']['y'] -= data_y[i_start:i_end].mean()*1e-12
        self.fc.invalidate_prodata_cache()
        if self.fc.data['tasktype']=='smfs':
            cal_baseline_x(self.fc)
        if allowRotate:
            self.fc.data['offset']['k'] = 0
            self.fc.invalidate_prodata_cache()
            data = self.fc.get_prodata()['retract']
            data_x,data_y = data['measuredHeight'],data['vDeflection']
            k = get_slope(data_x[i_start:i_end].reshape(-1),data_y[i_start:i_end].reshape(-1),0)
            self.fc.data['offset']['rotate_index'] = i_start
            self.fc.data['offset']['k']=k
            self.fc.invalidate_prodata_cache()
            data = self.fc.get_prodata()['retract']
            data_y = data['vDeflection']
            self.fc.data['offset']['y'] -= data_y[i_start:i_end].mean()
            self.fc.invalidate_prodata_cache()
        self.curve_change()
        self.fc.clean_force()
        self.zpo.changedforce(save=False)
        self.change_dic={}
    def pk_delete(self):
        if not self.state:
            return None
        self._push_undo()
        if len(self.fc.data['peakindex'])>0:
            del self.fc.data['peakindex'][self.forcepeak_index]
            del self.fc.data['k'][self.forcepeak_index]
        if self.tasktype == 'smfs' and len(self.fc.data['wlcarg'])>0 :
            del self.fc.data['wlcarg'][self.forcepeak_index]
            self.fc.recover_force(self.ljp)
            process_customize(self.fc,[11,12])
        self.pk_indexchange(-1)
        self.curve_change()
    def fc_delete(self):
        if not self.state:
            return None
        self._push_undo()
        self.forcepeak_index = 0
        self.fc.data['artificial_judge']=False
        self.curve_change()
    def lp_change(self,dlp=0,amply=0.1):
        if not self.state or len(self.fc.data['wlcarg'])==0 or dlp==0:
            return None
        self._push_undo()
        self.fc.data['arg'] = self.taskarg
        real_peakindex = np.argwhere(self.zpo[self.forcecurve_index]['peakindex']==self.fc.data['peakindex'][self.forcepeak_index])[0][0]
        if not self.fixlc_changelp:
            self.fc.data['wlcarg'][self.forcepeak_index]=(self.fc.data['wlcarg'][self.forcepeak_index][0],
                                                 self.zpo[self.forcecurve_index]['wlcarg'][real_peakindex][1]+amply*dlp)
        else:
            if self.coor_data[0]<0 or self.coor_data[1]<0:
                return None
            lp = self.zpo[self.forcecurve_index]['wlcarg'][real_peakindex][1]+amply*dlp
            lc = wlc2lc(float(self.coor_data[0]*1e-9),float(self.coor_data[1]*1e-12),float(lp*1e-9)).real*1e9
            self.fc.data['wlcarg'][self.forcepeak_index]=(lc,lp)
            
        self.fc.recover_force(self.ljp)
        process_customize(self.fc,[11,12])
        self.curve_change()
    def lc_change(self,dlc=0,amply=1):
        if not self.state or len(self.fc.data['wlcarg'])==0 or self.fixlc_changelp or dlc==0:
            return None
        self._push_undo()
        self.fc.data['arg'] = self.taskarg
        real_peakindex = np.argwhere(self.zpo[self.forcecurve_index]['peakindex']==self.fc.data['peakindex'][self.forcepeak_index])[0][0]
        self.fc.data['wlcarg'][self.forcepeak_index]=(self.zpo[self.forcecurve_index]['wlcarg'][real_peakindex][0]+amply*dlc,
                                                 self.fc.data['wlcarg'][self.forcepeak_index][1])
        self.fc.recover_force(self.ljp)
        process_customize(self.fc,[11,12])
        self.curve_change()
    def k_change(self,k=1,amply=1):
        if not self.state or len(self.fc.data['k'])==0 or k==0:
            return None
        self._push_undo()
        self.fc.data['arg'] = self.taskarg
        real_peakindex = np.argwhere(self.zpo[self.forcecurve_index]['peakindex']==self.fc.data['peakindex'][self.forcepeak_index])[0][0]
        self.fc.data['k'][self.forcepeak_index] = self.zpo[self.forcecurve_index]['k'][real_peakindex]+k*amply
        self.curve_change()
    def pv_change(self,value):
        if not self.state or len(self.fc.data['peakindex'])==0:
            None
        self._push_undo()
        self.fc.recover_force(self.ljp)
        res_index = self.fc.data['peakindex'][self.forcepeak_index]+value
        if res_index<=0 or res_index>=max(self.fc.data['peakindex']):
            return None
        if self.forcepeak_index+1<len(self.fc.data['peakindex']) and res_index>self.fc.data['peakindex'][self.forcepeak_index+1]:
            return None
        self.fc.data['peakindex'][self.forcepeak_index]=res_index
        if self.fc.data['tasktype']=='smfs':
            process_customize(self.fc,[8,10,11,12])
        elif self.fc.data['tasktype']=='cell_curve':
            process_customize(self.fc,[9,10])
        self.fc.clean_force()
        self.curve_change()
        self.zpo.changedforce(save=False)
        self.change_dic={}
    def reset(self):
        if not self.state:
            return None
        self._push_undo()
        self.fc.recover_force(self.ljp)
        self.fc.data['arg'] = self.taskarg
        
        if self.tasktype == 'smfs':
            process_customize(self.fc,[6,7,8,9,10,11,12])
        elif self.tasktype == 'cell_curve':
            process_customize(self.fc,[6,9,10])
        
        self.fc.data['artificial_judge'] = True
        self.fc.clean_force()
        self.curve_change()
        self.zpo.changedforce(save=False)
        self.change_dic={}
    def set_springconstant_override(self, value):
        if not self.state:
            return None
        self._push_undo()
        self.fc.recover_force(self.ljp)
        self.fc.data['springConstant_override'] = value
        self.fc.invalidate_prodata_cache()
        self.fc.data['arg'] = self.taskarg
        if self.tasktype == 'smfs':
            process_customize(self.fc,[6,7,8,9,10,11,12])
        elif self.tasktype == 'cell_curve':
            process_customize(self.fc,[6,9,10])
        self.fc.data['artificial_judge'] = True
        self.fc.clean_force()
        self.curve_change()
        self.zpo.changedforce(save=False)
        self.change_dic={}
    def set_springconstant_override_all(self, value):
        if not self.state:
            return None
        for i in range(len(self.zpo)):
            fc_tmp = forcecurve()
            fc_tmp.data = copy.deepcopy(self.zpo[i])
            fc_tmp.recover_force(self.ljp)
            fc_tmp.data['springConstant_override'] = value
            fc_tmp.invalidate_prodata_cache()
            fc_tmp.data['arg'] = self.taskarg
            if self.tasktype == 'smfs':
                process_customize(fc_tmp, [6, 7, 8, 9, 10, 11, 12])
            elif self.tasktype == 'cell_curve':
                process_customize(fc_tmp, [6, 9, 10])
            fc_tmp.data['artificial_judge'] = True
            fc_tmp.clean_force()
            self.zpo.changingforce(fc_tmp)
        self.fc.data = copy.deepcopy(self.zpo[self.forcecurve_index])
        self.zpo.changedforce(save=False)
        self.change_dic = {}

    def clear_springconstant_override(self):
        if not self.state:
            return None
        self._push_undo()
        self.fc.recover_force(self.ljp)
        self.fc.data['springConstant_override'] = None
        self.fc.invalidate_prodata_cache()
        self.fc.data['arg'] = self.taskarg
        if self.tasktype == 'smfs':
            process_customize(self.fc,[6,7,8,9,10,11,12])
        elif self.tasktype == 'cell_curve':
            process_customize(self.fc,[6,9,10])
        self.fc.data['artificial_judge'] = True
        self.fc.clean_force()
        self.curve_change()
        self.zpo.changedforce(save=False)
        self.change_dic={}
    def savechange(self,name,saveas=False):
        if not self.state:
            return None
        if not name or not name.strip():
            return False
        self.change_dic={}
        self.zpo.changedforce(name,saveas)
        self.dirty = False
    def changemark(self,mark):
        if not self.state or self.tasktype!='smfs':
            return None
        self._push_undo()
        if self.forcepeak_index<len(self.fc.data['mark']):
            self.fc.data['mark'][self.forcepeak_index]=mark
            self.curve_change()
    def changeClass(self,c='N'):
        if not self.state:
            return None
        self._push_undo()
        #return None
        self.fc.data['class']=c
        self.curve_change()
    def plot(self,F):
        if not self.state:
            return None
        if self.forcecurve_index in self.change_dic:
            datamsg = self.change_dic[self.forcecurve_index]
            if datamsg in self.zpo.change:
                self.fc.data = copy.deepcopy(self.zpo.change[datamsg])
            else:
                self.fc.data = copy.deepcopy(self.zpo[self.forcecurve_index])
        else:
            self.fc.data = copy.deepcopy(self.zpo[self.forcecurve_index])
        self.fc._prodata_cache.clear()
        self.fc._prodata_dirty = True
        include_last = self.taskarg.get('include_last', True)
        if 'arg' in self.fc.data.keys():
            self.taskarg = copy.deepcopy(self.fc.data['arg'])
        self.taskarg['include_last'] = include_last
        self.fc.data['arg'] = self.taskarg
        if self.forcepeak_index>len(self.fc.data['peakindex'])-1:
            self.forcepeak_index = len(self.fc.data['peakindex'])-1
        fc = copy.deepcopy(self.fc)
        F.plot(fc,self.forcepeak_index,self.ljp,self.tasktype,curve_index=self.forcecurve_index)
    def plot_contourhist(self):
        if not self.state or self.tasktype != 'smfs':
            return None
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_y = data['vDeflection']*1e12
        data_x = data['measuredHeight']*1e9
        img = Lc_transformer(data_x,data_y)
        return img
    def plot_contourscatter(self):
        if not self.state or self.tasktype != 'smfs':
            return None
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_y = data['vDeflection']*1e12
        data_x = data['measuredHeight']*1e9
        img = Lc_transformer(data_x,data_y,'scatter')
        return img
    def adhesionmap(self):
        if not self.state:
            return None
        arr = self.zpo.get_maxforce(self.ljp)
        img = plotmap(arr)
        return img
    def adhesionhist(self):
        if not self.state:
            return None
        arr = self.zpo.get_maxforce(self.ljp)
        img = plothist(arr)
        return img
    def drawlabel(self,label,lclplabel):
        if not self.state:
            return None
        label.setText('Peak select: {}/{}'.format(self.forcecurve_index,len(self.zpo)-1))
        try:
            self.fc.recover_force(self.ljp)
        except Exception:
            return
        data = self.fc.get_prodata()['retract']
        data_y = data['vDeflection']*1e12
        self.fc.clean_force()
        if self.tasktype!='smfs':
            if self.forcepeak_index < len(self.fc.data.get('peakindex', [])):
                lclplabel.setText(' F={:.2f}pN; k={:.1f}'.format(data_y[self.fc.data['peakindex'][self.forcepeak_index]][0],self.fc.data['k'][self.forcepeak_index]))
        else:
            pi = self.forcepeak_index
            peaks = self.fc.data.get('peakindex', [])
            wlc = self.fc.data.get('wlcarg', [])
            dlc_arr = self.fc.data.get('dlc', [])
            k_arr = self.fc.data.get('k', [])
            wq = self.fc.data.get('wlc_quality', [])
            we = self.fc.data.get('wlc_err', [])
            if pi >= len(peaks) or pi >= len(wlc):
                return
            bad = ''
            if pi < len(wq) and not wq[pi]:
                bad = ' !'
            elif pi < len(we) and we[pi][0] > 1.5:
                bad = ' !'
            try:
                if pi < len(peaks)-1 and pi < len(dlc_arr) and pi < len(k_arr):
                    lclplabel.setText(' Lc={:.1f}nm; F={:.2f}pN; dLc={:.1f}nm; k={:.1f}{}'.format(wlc[pi][0],data_y[peaks[pi]][0],dlc_arr[pi],k_arr[pi],bad))
                elif len(wlc) > 0:
                    lclplabel.setText(' Lc={:.1f}nm; F={:.2f}pN; k={:.1f}{}'.format(wlc[pi][0],data_y[peaks[pi]][0],k_arr[pi],bad))
            except (IndexError, KeyError):
                pass
    def copypeak_(self):
        if not self.state:
            return None
        if len(self.fc.data['peakindex'])==0:
            return None
        self._push_undo()
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_x,data_y = data['measuredHeight'].reshape(-1)*1e9,data['vDeflection'].reshape(-1)*1e12
        if self.forcepeak_index==0:
            index = np.where(data_x<data_x[self.fc.data['peakindex'][self.forcepeak_index]]-self.taskarg['xsens'])[0]
        else:
            index = np.where((data_x<data_x[self.fc.data['peakindex'][self.forcepeak_index]]-self.taskarg['xsens'])&(data_x>data_x[self.fc.data['peakindex'][self.forcepeak_index-1]]+self.taskarg['xsens']))[0]
        p = findpeak_smallrange(data_y[index])
        if len(p)==0:
            v=index[-1]-1
        else:
            v=index[0]+p[-1]
        if self.forcepeak_index>=0 and self.tasktype=='smfs':
            self.fc.data['peakindex'].insert(self.forcepeak_index,v)
            process_customize(self.fc,[8,10,11,12])
            self.fc.clean_force()
            self.curve_change()
            self.zpo.changedforce()
            self.change_dic={}
            self.dirty = False
        self.fc.clean_force()
    def copypeak(self,xdata,ydata):
        if not self.state:
            return None
        self._push_undo()
        self.fc.recover_force(self.ljp)
        data = self.fc.get_prodata()['retract']
        data_x,data_y = data['measuredHeight'].reshape(-1)*1e9,data['vDeflection'].reshape(-1)*1e12
        index = np.argmin(np.abs(data_x-xdata))
        i = len(np.where(self.fc.data['peakindex']<index)[0])
        if self.forcepeak_index>=0:
            self.fc.data['peakindex'].insert(i,index)
        if len(self.fc.data['peakindex'])==0:
            self.fc.data['peakindex'].append(i)
            self.forcepeak_index = 0
        if  self.fc.data['tasktype']=='smfs':
            process_customize(self.fc,[8,10,11,12])
        elif self.fc.data['tasktype']=='cell_curve':
            process_customize(self.fc,[9,10])
        self.fc.clean_force()
        self.curve_change()
        self.zpo.changedforce(save=False)
        self.change_dic={}
        self.forcepeak_index=i
        self.fc.clean_force()
    def keepcurve(self):
        self.fc.data['overlay'] = True
        self.curve_change()
    def discardcurve(self):
        self.fc.data['overlay'] = False
        self.curve_change()
    def export_prodata(self,sel):
        if not self.state:
            return None
        self.zpo.changedforce(save=False)
        T = None
        if self.tasktype == 'cell_curve':
            T = self.zpo.export_celldata(self.ljp,f_index=self.forcecurve_index)
        elif self.tasktype == 'smfs':
            try :
                include_last = self.taskarg.get('include_last', True)
                T = self.zpo.get_arg(self.ljp,f_index=self.forcecurve_index, include_last=include_last)
            except Exception as err:
                print(err)
        if not T:
            QMessageBox.information(sel,"Warning","Failed!")
    def exporttxt(self):
        if not self.state:
            return None
        if self.tasktype == 'cell_curve':
            self.zpo.exporttxt(self.ljp,self.forcecurve_index)
        elif self.tasktype == 'smfs':
            self.zpo.exporttxt(self.ljp,self.forcecurve_index)
    def exportbatchtxt(self):
        if not self.state:
            return None
        for i in range(self.forcecurve_index+1):
            self.zpo.exporttxt(self.ljp,i,tip_correc=False)
    def export_figure(self,figure):
        if not self.state:
            return None
        todir = os.path.dirname(self.zpo.fname)
        fname = os.path.join(todir,'{}.png'.format(self.forcecurve_index))
        figure.savefig(fname,bbox_inches='tight',transparent=True)
    def KNcluster(self,m_run):
        if not self.state or self.tasktype!='smfs':
            return None
        selfname = self.zpo.fname
        rawname = os.path.splitext(os.path.basename(selfname))[0]
        dirname = os.path.dirname(selfname)
        outname = os.path.join(dirname,'{}.cluster-matrix'.format(rawname))
        if not os.path.isfile(outname):
            matrix = get_distmatrix(self.zpo,self.ljp,m_run,length=400,step=2,thre=30)
            np.savetxt(outname,matrix)
        else:
            matrix = np.loadtxt(outname)
        index = list(KMsClustering(matrix,n_clusters=8))
        SplitDic = {}
        for i,class_index in enumerate(index):
            if class_index not in SplitDic.keys():
                SplitDic[class_index] = []
            else:
                SplitDic[class_index].append(i)
        self.zpo.split_DataYee(SplitDic)
    def offset_move(self,dx,dy,move=True):
        if not self.state:
            return None
        if move:
            self.fc.data['offset']['x']=self.zpo[self.forcecurve_index]['offset']['x']+dx*-1e-9
            self.fc.data['offset']['y']=self.zpo[self.forcecurve_index]['offset']['y']+dy*1e-12
        else:
            self.zpo[self.forcecurve_index]['offset']['x'] = self.fc.data['offset']['x']
            self.zpo[self.forcecurve_index]['offset']['y'] = self.fc.data['offset']['y']
        self.curve_change()
    def SimilaritySort(self,m_run):
        if not self.state or self.tasktype!='smfs':
            return None
        selfname = self.zpo.fname
        rawname = os.path.splitext(os.path.basename(selfname))[0]
        dirname = os.path.dirname(selfname)
        outname = os.path.join(dirname,'{}.cluster-matrix'.format(rawname))
        if not os.path.isfile(outname):
            matrix = get_distmatrix(self.zpo,self.ljp,m_run,length=400,step=2,thre=30)
            np.savetxt(outname,matrix)
        else:
            matrix = np.loadtxt(outname)
        index = list(sort_similar(self.forcecurve_index,matrix))
        SplitDic = {0:index}
        self.zpo.split_DataYee(SplitDic)
    def hist_scatterplot(self,fig,ax_s):
        pass
    def Data_equip_slim(self,s,progress,sel):
        if not self.state:
            return None
        progress.setWindowTitle("Please Wait")  
        progress.setLabelText("Processing...")
        progress.setCancelButtonText("Cancel")
        progress.setMinimumDuration(5)
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0,len(self.zpo))
        fc  =forcecurve()
        for i,d in enumerate(self.zpo):
            progress.setValue(i)
            if progress.wasCanceled():
                QMessageBox.warning(progress,"Warning!","Failed!")
                break
            fc.data = d
            if s=='e':
                fc.recover_force(self.ljp)
            elif s=='s':
                fc.data['compressed_data'] = {}
            fc.clean_force()
            self.zpo.changingforce(fc)
        self.zpo.changedforce()
        self.dirty = False
        progress.setValue(len(self.zpo))
        QMessageBox.information(sel,"Notic","Success")
    def execu_autostep(self,progress,sel):
        self.zpo.delet_dataYee()
        self.change_dic = {}
        self.highspeedcorr = np.array([])
        num = len(self.ljp)
        progress.setWindowTitle("Please Wait")  
        progress.setLabelText("Processing...")
        progress.setCancelButtonText("Cancel")
        progress.setMinimumDuration(5)
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0,num)
        source_path = self.ljp.filedir
        n_workers = min(os.cpu_count() or 1, 8, max(1, num // 20))
        if n_workers > 1 and num > 20:
            progress.setLabelText(f'Processing {num} curves ({n_workers} workers)...')
            try:
                from multiprocessing import Pool
                tasks = [(i, dict(self.taskarg), self.tasktype) for i in range(num)]
                chunk = max(1, num // (n_workers * 5))
                with Pool(n_workers, initializer=_worker_init, initargs=(source_path,)) as pool:
                    n_done = 0
                    for result in pool.imap_unordered(_worker_process, tasks, chunksize=chunk):
                        if progress.wasCanceled():
                            pool.terminate()
                            self.zpo.delet_dataYee()
                            self.ready_run = True
                            return
                        idx, fc_data, hs = result
                        if fc_data is not None:
                            fc = forcecurve()
                            fc.data = fc_data
                            self.zpo.changingforce(fc)
                            if self.taskarg.get('highspeed') and hs > 0:
                                self.highspeedcorr = np.append(self.highspeedcorr, hs)
                            print(idx, len(self.ljp), len(self.zpo))
                        n_done += 1
                        progress.setValue(n_done)
            except Exception:
                import traceback
                traceback.print_exc()
                progress.setLabelText('Multi-worker failed, fallback to serial...')
                self.highspeedcorr = np.array([])
                self.zpo.delet_dataYee()
                for i in range(num):
                    print(i, len(self.ljp), len(self.zpo))
                    progress.setValue(i)
                    if progress.wasCanceled():
                        self.zpo.delet_dataYee()
                        self.ready_run = True
                        return
                    self.fc.data = self.ljp[i]
                    self.fc._prodata_cache.clear()
                    self.fc._prodata_dirty = True
                    self.fc.data['arg'] = self.taskarg
                    main(self.fc, self.zpo, self.tasktype)
                    if self.taskarg['highspeed'] and self.fc.data['offset']['highspeed'] > 0:
                        self.highspeedcorr = np.append(self.highspeedcorr, self.fc.data['offset']['highspeed'])
        else:
            progress.setLabelText('Processing...')
            for i in range(num):
                print(i, len(self.ljp), len(self.zpo))
                progress.setValue(i)
                if progress.wasCanceled():
                    QMessageBox.warning(sel, "Warning!", "Failed!")
                    self.zpo.delet_dataYee()
                    self.ready_run = True
                    return
                self.fc.data = self.ljp[i]
                self.fc._prodata_cache.clear()
                self.fc._prodata_dirty = True
                self.fc.data['arg'] = self.taskarg
                main(self.fc, self.zpo, self.tasktype)
                if self.taskarg['highspeed'] and self.fc.data['offset']['highspeed'] > 0:
                    self.highspeedcorr = np.append(self.highspeedcorr, self.fc.data['offset']['highspeed'])
        if self.taskarg.get('highspeed') and len(self.highspeedcorr) > 0:
            avg_hs = self.highspeedcorr.mean()
            for k in self.zpo.change:
                self.zpo.change[k]['offset']['highspeed'] = avg_hs
        if len(self.zpo.change) == 0:
            self.state = False
        else:
            self.state = True
            self.zpo.changedforce()
            classes = []
            pkl = self.zpo.data['data.pkl']
            keys = list(pkl.keys())
            for k in keys:
                classes.append(_sort_key(pkl[k].get('class', 'N')))
            sorted_pairs = sorted(zip(classes, keys), key=lambda x: x[0])
            new_pkl = {}
            for _, k in sorted_pairs:
                new_pkl[k] = pkl[k]
            self.zpo.data['data.pkl'] = new_pkl
            self.zpo.savefile()
        self.dirty = False
        self.change_dic.clear()
        progress.setValue(num)
        QMessageBox.information(sel, "Notic", "Success")
        self.ready_run = False

    def train_and_auto_classify(self, progress=None, sel=None):
        if not self.state or self.tasktype != 'smfs':
            return None
        self.zpo.changedforce(save=False)
        if progress:
            progress.setLabelText('Learning from manual labels...')
            progress.setValue(5)
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        if progress:
            progress.setLabelText('Learning from manual labels...')
            progress.setValue(3)
            QApplication.processEvents()
        class_stats = learn_class_stats(self.zpo, self.ljp)
        if progress:
            progress.setLabelText('Extracting features from training curves...')
            progress.setValue(7)
            QApplication.processEvents()
        def _train_progress(pct):
            if progress:
                progress.setValue(pct)
                progress.setLabelText('Training SVM ({:.0f}%)...'.format(pct / 30.0 * 100))
                QApplication.processEvents()
        clf, scaler, classes_list, train_counts, trained_mark_defs = train_classifier(self.zpo, self.ljp, progress_cb=_train_progress)
        if clf is None:
            return 'Not enough labeled curves (need >=5, have >=2 classes)'
        train_info = ' '.join(f'{c}:{n}' for c, n in train_counts.items())
        stat_parts = []
        for c, s in class_stats.items():
            stat_parts.append(c + '(dLc={:.0f}±{:.0f}nm F={:.0f}pN)'.format(s['dLc_median'], s['dLc_std'], s['F_median']))
        stat_info = ' '.join(stat_parts)
        n_total = len(self.zpo)
        auto_targets = []
        manual_targets = []
        for i in range(n_total):
            d = self.zpo[i]
            if len(d.get('peakindex', [])) == 0:
                continue
            cls = d.get('class', 'N')
            if cls not in ('N', 'Z', 'F'):
                manual_targets.append(i)
            else:
                auto_targets.append(i)
        if progress:
            progress.setLabelText(f'Classifying {len(auto_targets)} F/N/Z curves...')
            progress.setValue(30)
            QApplication.processEvents()
        n_changed, n_done = 0, 0
        pred_counts = {}
        rescue_candidates = []
        n_rescued = 0
        for i in auto_targets:
            d = self.zpo[i]
            if progress and n_done % max(1, len(auto_targets) // 50) == 0:
                pct = 30 + int(50 * n_done / max(len(auto_targets), 1))
                progress.setValue(min(pct, 100))
                QApplication.processEvents()
                if progress.wasCanceled():
                    return 'Auto-Classify cancelled'
            fc_tmp = forcecurve()
            fc_tmp.data = d
            try:
                if not d.get('peak_forces'):
                    fc_tmp.recover_force(self.ljp)
                pred, _ = predict_curve_class(fc_tmp, clf, scaler, trained_mark_defs)
                fc_tmp.clean_force()
            except Exception:
                pred = d.get('class', 'N')
            old_cls = d.get('class', 'N')
            if pred != old_cls:
                n_changed += 1
            fc_tmp.data['class'] = pred
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
            self.zpo.changingforce(fc_tmp)
            n_done += 1
            good_classes = [c for c in train_counts if c != 'F']
            if pred == 'F' and good_classes and class_stats:
                for gc in good_classes:
                    if gc not in class_stats:
                        continue
                    stat = class_stats[gc]
                    n_peaks = len(d.get('peakindex', []))
                    dlc_vals = d.get('dlc', [])
                    pn_ok = stat['peak_n_min'] <= n_peaks <= stat['peak_n_max']
                    dlc_match = any(stat['dLc_min'] < v < stat['dLc_max'] for v in dlc_vals) if len(dlc_vals) > 0 else False
                    if pn_ok and dlc_match:
                        rescue_candidates.append((i, gc))
                        break
        n_rescued = 0
        if rescue_candidates:
            n_rescued = 0
            if progress:
                progress.setLabelText(f'Rescue re-analysis: {len(rescue_candidates)} candidates...')
                progress.setValue(80)
                QApplication.processEvents()
            for j, (idx, target_cls) in enumerate(rescue_candidates):
                if progress and j % max(1, len(rescue_candidates) // 20) == 0:
                    pct = 80 + int(19 * j / max(len(rescue_candidates), 1))
                    progress.setValue(min(pct, 100))
                    QApplication.processEvents()
                d = self.zpo[idx]
                fc_tmp = forcecurve()
                fc_tmp.data = d
                try:
                    if not d.get('peak_forces'):
                        fc_tmp.recover_force(self.ljp)
                    if target_cls in class_stats:
                        rescue_reanalyze(fc_tmp, class_stats, target_cls)
                    new_pred, _ = predict_curve_class(fc_tmp, clf, scaler, trained_mark_defs)
                except Exception:
                    new_pred = d.get('class', 'N')
                fc_tmp.clean_force()
                old_cls_2 = d.get('class', 'N')
                if new_pred != old_cls_2:
                    fc_tmp.data['class'] = new_pred
                    n_rescued += 1
                    pred_counts[new_pred] = pred_counts.get(new_pred, 0) + 1
                    pred_counts[old_cls_2] = max(0, pred_counts.get(old_cls_2, 1) - 1)
                else:
                    fc_tmp.data["class"] = old_cls_2
                self.zpo.changingforce(fc_tmp)
            if progress:
                progress.setValue(95)
        self.dirty = True
        self.zpo.changedforce(save=False)
        self.change_dic.clear()
        pkl = self.zpo.data['data.pkl']
        keys = list(pkl.keys())
        classes = [_sort_key(pkl[k].get('class', 'N')) for k in keys]
        sorted_pairs = sorted(zip(classes, keys), key=lambda x: x[0])
        new_pkl = {}
        for _, k in sorted_pairs:
            new_pkl[k] = pkl[k]
        self.zpo.data['data.pkl'] = new_pkl
        if progress:
            progress.setValue(100)
        pred_info = ' '.join(f'{c}:{n}' for c, n in sorted(pred_counts.items()))
        return f'{stat_info} → Auto [{pred_info}] changed={n_changed} rescued={n_rescued} | Manual {len(manual_targets)} preserved'

if __name__ == '__main__':
    import time,datetime
    t1 = time.time()
    #batch_fc_pro(r'D:\code\py\DataYeeN\20201125-COH-I30-I32-NGL-9GLINKER\20201125-COH-I30-I32-NGL-9GLINKER')
    t2 = time.time()
    print(str(datetime.timedelta(seconds=t2-t1)))

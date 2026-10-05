# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 22:19:46 2022

@author: zbin
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, find_peaks
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from src.loadjpk import forcecurve,zipfileopera
import pickle
import zipfile
from zipfile import ZipFile
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
def lcfunc(x, lc, lp):
    return 1.3806e-23 * 298 / (lp * 1e-9) * (1 / 4 * (1 - x / lc) ** (-2) + x / lc - 1 / 4) * 1e12
T = 298
kb = 1.38e-23
def BE(x_arr,x_beta,k_off):
    f_beta = kb*T/x_beta
    F_arr = f_beta*np.log(x_arr/f_beta/k_off)
    return F_arr
def gauss(x,mean,sigma):
    y = 1/sigma/np.sqrt(np.pi*2)*np.exp(-1*(x-mean)**2/sigma**2/2)
    return y

def linecut(a,b,c):
    a = a[np.where(b<c)]
    b = b[np.where(b<c)]
    return a,b

def copylist(item,n):
    if type(item)!=list:
        item = [item]*n
    elif len(item)<n:
        item+=[item[-1]]*(n-len(item))
    elif len(item)>n:
        item = item[:n]
    return item

def color_correct(c):
    if type(c)==str:
        c = (c,1)
    return c

def plot_forcecurve(ax,fc,color='k',lw=2,alpha=0.7,mark_color={},star_mark=[],star_type='*',star_size=30,smooth=9):
    fc.recover_force()
    data = fc.get_prodata(s=smooth)['retract']
    x,y = data['measuredHeight']*1e9,data['vDeflection']*1e12
    fc.clean_force()
    ax.plot(x,y,c=color,lw=lw,alpha=alpha)
    ##
    mark = fc.data['mark']
    p_i = fc.data['peakindex']
    for i,m in enumerate(mark):
        if m not in mark_color.keys():
            continue
        x_,y_ = x[p_i[i]:p_i[i+1]],y[p_i[i]:p_i[i+1]]
        ax.plot(x_,y_,lw=lw,color=mpl.colors.to_rgba(*color_correct(mark_color[m])))
        if m not in star_mark:
            continue
        x_,y_ = x[p_i[i]]-6,y[p_i[i]]+30
        ax.scatter(x_,y_,s=star_size,marker=star_type,c=color_correct(mark_color[m])[0])
    return ax

def plot_wlccurve(ax,wlcarg,ls='--',lw=2,color=[('r',1)],xcut=-1,ycut=-1):
    n = len(wlcarg)
    ls,lw,color,xcut,ycut = copylist(ls,n),copylist(lw,n),copylist(color,n),\
        copylist(xcut,n),copylist(ycut,n)
    for i,arg in enumerate(wlcarg):
        x = np.linspace(0,arg[0]-0.1,200)
        y = lcfunc(x,*arg)
        if xcut[i]>0:
            y,x = linecut(y,x,xcut[i])
        if ycut[i]>0:
            x,y = linecut(x,y,ycut[i])
        ax.plot(x,y,lw=lw[i],color=mpl.colors.to_rgba(*color_correct(color[i])),ls=ls[i])
    return ax

def plot_hist(ax,data,bins=20,ranges=(0,200),lw=2,ls='--',ec='k',fc=('r',0.3),orientation='vertical',density=True):
    if 'v' in orientation.lower():
        orientation = 'vertical'
    elif 'h' in orientation.lower():
        orientation = 'horizontal'
    ax.hist(data,bins=bins,range=ranges,ec=mpl.colors.to_rgba(*color_correct(ec)),fc=mpl.colors.to_rgba(*color_correct(fc)),\
            orientation=orientation,density=density,lw=lw)
    return ax

def plot_hist_curve(ax,data,bins=20,ranges=(0,200),lw=2,ls='--',color='k',density=True,orientation='vertical',curvetype='gaus'):
    x = np.linspace(ranges[0],ranges[1],400)
    if 'g' in curvetype.lower():
        a = np.histogram(data,range=ranges,bins=bins,density=True)
        kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(data.reshape(-1,1))
        log_dens = kde.score_samples(x.reshape(-1,1))
        p,_ = find_peaks(np.exp(log_dens)/np.exp(log_dens).max(),height=0.15,distance=5)
        force_kde = x[p[0]]
        x_,y_ = a[1][:-1]+np.diff(a[1]).mean()*0.5,a[0]
        popt,pocv = curve_fit(gauss,x_,y_,p0=[30,force_kde],bounds=([0,0],[1000,1000]))
    elif 'm' in curvetype.lower():
        popt = (data.mean(),data.std()) 
    x-=0
    y = gauss(x,*popt)
    if 'v' in orientation.lower():
        x,y = x,y
    elif 'h' in orientation.lower():
        x,y = y,x
    if not density:
        y = y*np.histogram(data,range=ranges,bins=bins,density=False)[0].max()/a[0].max()
    ax.plot(x,y,lw=lw,color=color,ls=ls)
    return ax,popt
def get_gausdata(data,bins=20,ranges=(0,200),density=True):
    x = np.linspace(ranges[0],ranges[1],400)
    a = np.histogram(data,range=ranges,bins=bins,density=density)
    kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(data.reshape(-1,1))
    log_dens = kde.score_samples(x.reshape(-1,1))
    p,_ = find_peaks(np.exp(log_dens)/np.exp(log_dens).max(),height=0.15,distance=5)
    force_kde = x[p[0]]
    x_,y_ = a[1][:-1]+np.diff(a[1]).mean()*0.5,a[0]
    popt,pocv = curve_fit(gauss,x_,y_,p0=[100,force_kde],bounds=([0,0],[1000,1000]))
    return popt
def get_lrdata(zpo_lst,mark,speed,clas):
    f_arr = np.array([])
    lr_arr = np.array([])
    for i,zpo in enumerate(zpo_lst):
        if type(zpo)==str:
            zpo = zipfileopera(zpo)
        dic = zpo.get_prodata(classes=[clas])
        if mark == 'Rupforce':
            force_mark = np.array(dic[mark][clas]['force'])
            lr = np.array(dic[mark][clas]['k'])
        else:
            force_mark = np.array(dic[mark]['force'])
            lr = np.array(dic[mark]['k'])
        f_arr = np.append(f_arr,force_mark)
        lr_arr = np.append(lr_arr,lr)
    return np.dstack((lr_arr,f_arr))[0]
def Bell_Evans_fit(force,lr,x_b=-1):
    import numpy as np
    from scipy.optimize import curve_fit
    kbT = 1.38e-23*298
    def BE(x_arr,x_beta,k_off):
        f_beta = kbT/x_beta
        F_arr = f_beta*np.log(x_arr/f_beta/k_off)
        return F_arr
    if x_b <=0:
        a,b = np.polyfit(np.log(lr),force,1)
        x_b = kbT/a
        k_off = x_b/(kbT*np.exp(b*x_b/kbT))
    else:
        def line(x,b):
            return kbT/x_b*x+b
        popt,pcov = curve_fit(line,np.log(lr),force)
        b = popt[0]
        k_off = x_b/(kbT*np.exp(b*x_b/kbT))
    popt,pcov = curve_fit(BE,lr,force,p0=(x_b,k_off),bounds=([x_b,k_off],[x_b+x_b*0.01,k_off+k_off*0.01]))
    perr = np.sqrt(np.diag(pcov))
    result = {'x_b':[x_b,perr[0]],'k_off':[k_off,perr[1]]}
    return result
def ax_setallnone(ax):
    ax.set_facecolor('none')
    ax.set_yticks([])
    ax.set_xticks([])
    [ax.spines[s].set_visible(False) for s in ['left','bottom']]
    return ax
def save(fname,obj):
    with ZipFile(fname, 'w', zipfile.ZIP_DEFLATED) as zips:
        zips.writestr('plotdata', pickle.dumps(obj))
def open_(fname):
    with ZipFile(fname, 'r', zipfile.ZIP_DEFLATED) as zips:
        with zips.open('plotdata') as f:
            return pickle.load(f)
def savefig(fname,fig):
    fig.savefig(fname,bbox_inches='tight',transparent=True)
class forcecurve_fig():
    def __init__(self):
        self.plotdata = {"srcdata":[],
                         "rawdata":[],
                         'figsize':(8,6),
                         'dpi':300,
                         'xlim':(-5,300),
                         'ylim':(-60,500),
                         'xticks':[],
                         'yticks':[],
                         'forceline':{},
                         'wlcline':{},
                         'hspace':0.1}
        self.fc = forcecurve()
        self.set_forceparams()
        self.set_wlcparams()
        self.axs = []
    def appenddata(self,zpofname,index_lst):
        self.plotdata['srcdata'].append((zpofname,index_lst))
    def loaddata(self):
        self.plotdata['rawdata'] = []
        fc = self.fc
        for zpofname,index_lst in self.plotdata['srcdata']:
            zpo = zipfileopera(zpofname)
            for i in index_lst:
                fc.data = zpo[i]
                self.plotdata['rawdata'].append((zpofname,i,fc.data))
    def open_(self,fname):
        self.plotdata = open_(fname)
    def save(self,fname):
        save(fname,self.plotdata)
    def savefig(self,fname):
        savefig(fname, self.fig)
    def clean_axes(self):
        [plt.delaxes(ax=ax) for ax in self.axs]
    def create_axes(self,n,ax):
        if 'hspace' not in self.plotdata.keys():
            self.plotdata['hspace'] = 0.05
        space = self.plotdata['hspace']
        axs = [ax.inset_axes([0,1-1/n*(i+1),1,1/n-space]) for i in range(n)]
        [ax.set_facecolor('none') for ax in axs]
        self.axs = axs
    def quickplot(self):
        fc = self.fc
        self.create_fig(len(self.plotdata['rawdata']),1)
        self.set_displace()
        for i,item in enumerate(self.axs):
            _,_,d = self.plotdata['rawdata'][i]
            fc.data = d
            ax  = self.axs[i]
            ax  = plot_forcecurve(ax, fc,**self.plotdata['forceline'])
            ax_ = ax.inset_axes([0,0,1,1.2])
            ax_ = ax_setallnone(ax_)
            ax_.set_xlim(self.plotdata['xlim'])
            ax_.set_ylim([-50,340])
            ax_ = plot_wlccurve(ax_, fc.data['wlcarg'],**self.plotdata['wlcline'])
    def plot(self):
        fc = self.fc
        self.set_displace()
        for i,item in enumerate(self.axs):
            _,_,d = self.plotdata['rawdata'][i]
            fc.data = d
            ax  = self.axs[i]
            ax  = plot_forcecurve(ax, fc,**self.plotdata['forceline'])
            ax_ = ax.inset_axes([0,0,1,1.2])
            ax_ = ax_setallnone(ax_)
            ax_.set_xlim(self.plotdata['xlim'])
            ax_.set_ylim([-50,340])
            ax_ = plot_wlccurve(ax_, fc.data['wlcarg'],**self.plotdata['wlcline'])
    def set_forceparams(self,color='k',lw=2,alpha=0.7,mark_color={},star_mark=[],star_type='*',star_size=30,smooth=9):
        self.plotdata['forceline'] = dict(color=color,lw=lw,alpha=alpha,mark_color=mark_color,star_mark=star_mark,\
                                          star_type=star_type,star_size=star_size,smooth=smooth)
    def set_wlcparams(self,ls='--',lw=2,color=[('r',1)],xcut=-1,ycut=-1):
        self.plotdata['wlcline'] = dict(ls=ls,lw=lw,color=color,xcut=xcut,ycut=ycut)
    def set_displace(self):
        for i,ax in enumerate(self.axs):
            ax.set_xlim(self.plotdata['xlim'])
            ax.set_ylim(self.plotdata['ylim'])
            ax.set_yticks(self.plotdata['yticks'])
            ax.spines['left'].set_visible(False)
            if i<len(self.axs)-1:
                ax.set_xticks([])
                ax.spines['bottom'].set_linewidth(0.5)
            else:
                ax.set_xticks(self.plotdata['xticks'])
                ax.set_xlabel('Extension (nm)')
                ax.plot([10,10],[100,200],lw=2,color='k',alpha=0.8)

class loadingrate_fig():
    def __init__(self):
        self.plotdata = {
            "srcdata":{
                },
            "rawdata":{
                },
            "scatterparams":{
                },
            'dataponitparams':{
                },
            "fitlineparams":{
                },
            "errorlineparams":{
                },
            "histparams":{
                },
            "histlineparams":{
                },
            'hspace':0.1
            }
    def open_(self,fname):
        self.plotdata = open_(fname)
    def save(self,fname):
        save(fname,self.plotdata)
    def appenddata(self,zpofname,expname,clas,speed,mark):
        if expname not in self.plotdata['srcdata'].keys():
            self.plotdata['srcdata'][expname] = {}
        if speed not in self.plotdata['srcdata'][expname].keys():
            self.plotdata['srcdata'][expname][speed] = {}
        if 'zpofname' not in self.plotdata['srcdata'][expname][speed].keys():
            self.plotdata['srcdata'][expname][speed]['zpofname'] = []
        if 'clas' not in self.plotdata['srcdata'][expname].keys():
            self.plotdata['srcdata'][expname]['clas'] = clas
        if 'mark' not in self.plotdata['srcdata'][expname].keys():
            self.plotdata['srcdata'][expname]['mark'] = mark
        self.plotdata['srcdata'][expname][speed]['zpofname'].append(zpofname)
    def loaddata(self):
        self.plotdata['rawdata'] = {}
        dic_temp = {}
        for expname,v in self.plotdata['srcdata'].items():
            if expname not in self.plotdata['rawdata'].keys():
                self.plotdata['rawdata'][expname] = {}
            for k,vv in v.items():
                if k == 'clas':
                    self.plotdata['rawdata'][expname]['clas'] = vv
                elif k == 'mark':
                    self.plotdata['rawdata'][expname]['mark'] = vv
                elif type(k) == int:
                    speed = k
                    mark = self.plotdata['srcdata'][expname]['mark']
                    clas = self.plotdata['srcdata'][expname]['clas']
                    self.plotdata['rawdata'][expname][speed] = {}
                    self.plotdata['rawdata'][expname][speed]['datapoint'] \
                        = get_lrdata(vv['zpofname'],mark,speed,clas)
                    self.plotdata['rawdata'][expname][speed]['data']\
                        = get_gausdata(self.plotdata['rawdata'][expname][speed]['datapoint'][:,0]),\
                            get_gausdata(self.plotdata['rawdata'][expname][speed]['datapoint'][:,1])
    def re_cala(self):
        for expname,v in self.plotdata['srcdata'].items():
            for k,vv in v.items():
                if type(k) == int:
                    speed = k
                    self.plotdata['rawdata'][expname][speed]['data']\
                        = get_gausdata(self.plotdata['rawdata'][expname][speed]['datapoint'][:,0]),\
                            get_gausdata(self.plotdata['rawdata'][expname][speed]['datapoint'][:,1])
    def create_fig(self):
        self.fig = plt.figure(figsize=(16,6),dpi=300)
        plt.subplots_adjust(wspace=10)
        gs = self.fig.add_gridspec(20,20)
        self.ax0 = self.fig.add_subplot(gs[:,:10])
        self.ax0.set_xscale('log')
        self.ax0.set_ylim([0,300])
        self.ax0.set_xlim([0.2e4,1.1e6])
        ax1 = self.fig.add_subplot(gs[:,10:12])
        ax2 = self.fig.add_subplot(gs[:,12:14])
        ax3 = self.fig.add_subplot(gs[:,14:16])
        ax4 = self.fig.add_subplot(gs[:,16:18])
        ax5 = self.fig.add_subplot(gs[:,18:20])
        self.axs = [ax1,ax2,ax3,ax4,ax5]
        [ax.set_ylim([0,300]) for ax in self.axs]
        [ax.set_yticks([]) for ax in self.axs]
        [ax.set_xticks([]) for ax in self.axs]
    def quickplot(self):
        self.create_fig()
        for expname,v in self.plotdata['rawdata'].items():
            f,l = [],[]
            n=0
            for k,vv in v.items():
                if k in ['mark','clas','n']:
                    continue
                elif type(k) == int:
                    ax = self.axs[n]
                    speed = k
                    x,y = vv['datapoint'][:,0]*speed,vv['datapoint'][:,1]
                    self.ax0.scatter(x,y,**self.plotdata['scatterparams'][expname])
                    plot_hist(ax,y,**self.plotdata['histparams'][expname])
                    plot_hist_curve(ax,y,**self.plotdata['histlineparams'][expname])
                    x,y = vv['data'][0][0]*speed,vv['data'][1][0]
                    yerr = vv['data'][1][1]
                    self.ax0.scatter(x,y,**self.plotdata['dataponitparams'][expname])
                    f.append(y),l.append(x)
                    self.ax0.errorbar(x,y,yerr=yerr,**self.plotdata['errorlineparams'][expname])
                    n+=1
            f,l = np.array(f)*1e-12,np.array(l)*1e-12
            print(f,l)
            result = Bell_Evans_fit(f,l,x_b=-1)
            x_b,x_b_std = result['x_b']
            k_off,k_off_std = result['k_off']
            x_  = np.linspace(4e3,6e5,1000)
            y_ = BE(x_*1e-12,x_b,k_off)*1e12
            print(expname,result)
            self.ax0.plot(x_,y_,label=expname,**self.plotdata['fitlineparams'][expname])
    def set_scatterparams(self,expname,s=4,marker='o',alpha=0.3,fc='none',color='k'):
        self.plotdata['scatterparams'][expname] = dict(s=s,marker=marker,alpha=alpha,fc=fc,color=color)
    def set_datapointparams(self,expname,s=30,marker='s',alpha=0.9,fc='r',color='r'):
        self.plotdata['dataponitparams'][expname] = dict(s=s,marker=marker,alpha=alpha,fc=fc,color=color)
    def set_fitlineparams(self,expname,color='k',alpha=1,lw=2.5):
        self.plotdata['fitlineparams'][expname] = dict(color=color,alpha=alpha,lw=lw)
    def set_errorlineparams(self,expname,color='k',alpha=0.8,lw=1,capsize=3):
        self.plotdata['errorlineparams'][expname] = dict(color=color,alpha=alpha,lw=lw,capsize=capsize)
    def set_histparams(self,expname,bins=20,ranges=(0,200),lw=2,ls='--',ec='k',fc=('r',0.3),orientation='vertical',density=True):
        if 'histparams' not in self.plotdata.keys():
            self.plotdata['histparams']={}
        self.plotdata['histparams'][expname] = dict(bins=20,ranges=(0,200),lw=lw,ls=ls,ec=ec,fc=fc,orientation=orientation,density=density)
    def set_histlineparams(self,expname,bins=20,ranges=(0,200),lw=2,ls='--',color='k',density=True,orientation='vertical',curvetype='gaus'):
        if 'histlineparams' not in self.plotdata.keys():
            self.plotdata['histlineparams']={}
        self.plotdata['histlineparams'][expname] = dict(bins=bins,ranges=ranges,lw=lw,ls=ls,color=color,density=density,orientation=orientation,curvetype=curvetype)
    def savefig(self,fname):
        savefig(fname, self.fig)

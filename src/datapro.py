 # -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:51:44 2021

@author: ZhengBin
"""
import sys
sys.path.append('../')
import numpy as np
import copy
import warnings
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
from src.predictcore import feature_extract,MobileNet,myNet,Net
import matplotlib as mpl
from sklearn.neighbors import KernelDensity
from src.tool import timer
from src.loadjpk import forcecurve
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#m = MobileNet()
my = myNet('img')
plt.ioff()
def lcfunc(x, lc, lp):
    return 1.3806e-23 * 298 / (lp * 1e-9) * (1 / 4 * (1 - x / lc) ** (-2) + x / lc - 1 / 4) * 1e12
def wlc2lc(x,f,lp):
    kb = 1.3806e-23
    T = 298
    lc = (4*f*lp*x+3*kb*T*x)/(6*f*lp)-(-16*f**2*lp**2*x**2+12*f*kb*lp*T*x**2-36*kb**2*T**2*x**2)/(12*2**(2/3)*f*lp*(-16*f**3*lp**3*x**3+72*f**2*kb*lp**2*T*x**3-27*f*kb**2*lp*T**2*x**3+54*kb**3*T**3*x**3+3*(3)**0.5*(-64*f**5*kb*lp**5*T*x**6+144*f**4*kb**2*lp**4*T**2*x**6-108*f**3*kb**3*lp**3*T**3*x**6+135*f**2*kb**4*lp**2*T**4*x**6)**0.5)**(1/3))+(-16*f**3*lp**3*x**3+72*f**2*kb*lp**2*T*x**3-27*f*kb**2*lp*T**2*x**3+54*kb**3*T**3*x**3+3*(3)**0.5*(-64*f**5*kb*lp**5*T*x**6+144*f**4*kb**2*lp**4*T**2*x**6-108*f**3*kb**3*lp**3*T**3*x**6+135*f**2*kb**4*lp**2*T**4*x**6)**0.5)**(1/3)/(6*2**(1/3)*f*lp)
    return lc
def lcfunc1d(x,lc,lp):
    return 4.114188*(1/lc + 0.5/(lc*(1 - x/lc)**3))/lp
def rotate(data_x, data_y, index, k):
    theta = np.arctan(k) * -1
    return (data_x - data_x[index]) * np.sin(theta) + (data_y - data_y[index]) * np.cos(theta) + data_y[index]
def get_slope(x_arr,y_arr,index=-1):
    p = np.polyfit(x_arr,y_arr,1)
    d = np.polyder(p)
    k = np.polyval(d, x_arr[index])
    return k
def fig2img(fig):
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return img
def wlc_k_f_nox(lc,k):
    lp = 0.36
    x = (-1.028547**6*lc+250000*k*lc**2*lp)/(-1.028547**6+250000*k*lc*lp)+(80.11823662350369*(-1.057908931209**12*lc**3+5.142735**11*k*lc**4*lp-6.25**10*k**2*lc**5*lp**2)**(1/3))/(-1.028547**6+250000*k*lc*lp)
    f = 1.3806e-23 * 298 / (lp * 1e-9) * (1 / 4 * (1 - x / lc) ** (-2) + x / lc - 1 / 4) * 1e12
    return f
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
def noise_down(fc):
    data_y = fc.data['rawdata']['retract']['vDeflection'] * 1e12
    r = 0.9
    data_y_right = data_y[:, 0][int(r * len(data_y)):]
    data_y_right_smth = gaussian_filter(data_y_right, 21)
    for s in np.arange(17)[3::2]:
        err = np.abs(savgol_filter(data_y[:, 0][int(r * len(data_y)):], s, 2) - data_y_right_smth).mean()
        if err < 4:
            break
    if fc.data['arg'].get('sg_win_lens', 0) > 0:
        s = fc.data['arg']['sg_win_lens']
    fc.data['filters']['win_lens'] = s
    fc.invalidate_prodata_cache()
def cal_baseline_drift(fc):
    fc.data['offset']['k'] = 0
    data = fc.get_prodata()['retract']
    data_x,data_y = data['measuredHeight'],data['vDeflection']
    if len(fc.data['peakindex'])==0:
        index = int(len(data_x)*0.9)
        k = get_slope(data_x[index:].reshape(-1), data_y[index:].reshape(-1))
    else:
        k=0
    fc.data['offset']['k'] = k
    fc.invalidate_prodata_cache()
def cal_baseline_y(fc):
    fc.data['offset']['y'] = 0
    data = fc.get_prodata()['retract']
    data_y = data['vDeflection']
    index = int(len(data_y)*0.9)
    fc.data['offset']['y'] = data_y[index:].mean()*-1
    fc.invalidate_prodata_cache()
    
def cal_baseline_x(fc):
    fc.data['offset']['x'] = 0
    data = fc.get_prodata()['retract']
    data_x,data_y = data['measuredHeight'],data['vDeflection']
    fc.data['offset']['x'] = data_x.reshape(-1)[0]
    fc.invalidate_prodata_cache()
    '''
    for i, v in enumerate(data_y):
        if v * data_y[i + 1] < 0:
            fc.data['offset']['x'] = 0.5 * (data_x[i] + data_x[i + 1])
            break
        if i>int(0.5*len(data_x)):
            fc.data['offset']['x'] = data_x[0]
            break
    '''
def cal_highspeed_drift(fc):
    fc.data['offset']['highspeed'] = 0
    if not fc.data['arg']['highspeed']:
        return None
    data = fc.get_prodata()
    data_retract_y,data_extend_y = data['retract']['vDeflection'],data['extend']['vDeflection']
    data_retract_x,data_extend_x = data['retract']['measuredHeight'],data['extend']['measuredHeight']
    retract_index,extend_index = int(0.9*len(data_retract_y)),int(0.1*len(data_extend_y))
    if data_extend_x[0] > data_retract_x[retract_index]:
        corr = data_extend_y[:extend_index].mean()-data_retract_y[retract_index:].mean()
    else:
        corr = 0
    if corr>40e-12 or corr<0:
        return None
    fc.data['offset']['highspeed'] = 0.5*corr
    pass
def predict(fc):
    usemodel = fc.data['arg']['fastmode']
    global my
    if my.datatype!=usemodel:
        my = myNet(usemodel)
    if usemodel == 'img':
        fig = feature_extract(fc)
        img = fig2img(fig)
        data = img
    elif usemodel == 'series':
        data = np.vstack(feature_extract(fc,get_data=True))
        data = data.reshape((1,*data.shape)).astype(np.float32)
    score = my.predict(data)
    if fc.data['arg']['modelstrict']:
        v = 1
    else:
        v = 2
    if score<=v:
        fc.data['mobilenet_judge']=True
    else:
        fc.data['mobilenet_judge']=False

def findpeak(fc):
    fc.data['peakindex'] = np.array([])
    fc.data['bottomindex'] = np.array([])
    data = fc.get_prodata(tip_correc=False)['retract']
    data_x,data_y = data['measuredHeight']*1e9,data['vDeflection']*1e12
    d = np.gradient(np.gradient(gaussian_filter(data_y[:, 0], 13)))
    #extract noise index
    index_noise = int(len(d)*0.9)
    noise = np.abs(d[index_noise:]).max()*1.1
    #xlim index
    xlim_index = np.where(data_x>fc.data['arg']['xlim'])[0]
    if len(xlim_index) == 0:
        fc.data['peakindex'] = []
        fc.data['bottomindex'] = []
        fc.data['peak_forces'] = []
        return
    index_xlim = xlim_index[0]
    d = d[index_xlim:]
    distance = len(data_x) - np.where(data_x<data_x[-1]-fc.data['arg']['xsens'])[0][-1]
    p = find_peaks(d*-1,height=noise,distance=distance)[0]+index_xlim
    b = find_peaks(d,height=noise,distance=distance)[0]+index_xlim
    for p_i in p:
        temp_array = data_x[b] - data_x[p_i]
        i = np.where(temp_array > 0, temp_array, np.inf)
        if len(i)==0:
            continue
        b_i = b[i.argmin()]
        if p_i<b_i:
            if len(fc.data['peakindex'])!=0 and data_x[int(p_i)]-data_x[int(fc.data['peakindex'][-1])]<fc.data['arg']['xsens']:
                continue
            y = rotate(data_x[p_i - distance:b_i], data_y[p_i - distance:b_i], distance, -0.07)
            p_i = p_i - distance + np.argmax(y)
            k = np.polyval(np.polyder(np.polyfit(data_x[b_i:b_i + 300][:, 0], data_y[b_i:b_i + 300][:, 0], 1)), data_x[b_i])
            y = rotate(data_x[p_i:b_i + distance], data_y[p_i:b_i + distance], b_i - p_i, k - 0.07)
            b_i = p_i + np.argmin(y)
            fsens = data_y[p_i]-data_y[b_i]
            if fsens > fc.data['arg']['sens']:
                fc.data['peakindex'] = np.append(fc.data['peakindex'],p_i)
                fc.data['bottomindex'] = np.append(fc.data['bottomindex'],b_i)
    fc.data['peakindex'] = list(fc.data['peakindex'].astype(np.uint16))
    fc.data['bottomindex'] = list(fc.data['bottomindex'].astype(np.uint16))
    fc.data['peak_forces'] = [float(data_y[int(p_i)].reshape(-1)[0]) for p_i in fc.data['peakindex']]
    pass
def wlcfit(fc):
    fc.data['wlcarg'] = []
    fc.data['slopepre'] = []
    fc.data['wlc_quality'] = []
    fc.data['wlc_err'] = []
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
            noise_pre = np.std(np.diff(data_y[max(b_i,0):b_i+50])) if b_i+50 < p_i else 1.0
            if noise_pre < 1.5:
                ratio = 0.85
            elif noise_pre < 4.0:
                ratio = 0.70
            else:
                ratio = 0.55
            fitpoint = np.where(data_y[b_i:]>data_y[b_i]+ratio*dy)[0][0]+b_i
        else:
            fitpoint = p_i
        if fitpoint==b_i:
            fitpoint+=20
        x_fit = data_x[b_i:fitpoint]
        y_fit = data_y[b_i:fitpoint]
        # --- Physics-augmented multi-start WLC fitting ---
        try:
            _, x_wrc = WRC_transformer(y_fit, x_fit, thr=5)
            lc_wrc_vals = x_wrc[(x_wrc > data_x[p_i]) & (x_wrc < data_x[p_i] + 200)]
            if len(lc_wrc_vals) > 5:
                lc_median = float(np.median(lc_wrc_vals))
                lc_p25 = float(np.percentile(lc_wrc_vals, 25))
                lc_p75 = float(np.percentile(lc_wrc_vals, 75))
                lc_std = max(float(np.std(lc_wrc_vals)), 1.0)
            else:
                lc_median = data_x[p_i] + 25
                lc_p25 = lc_median - 5
                lc_p75 = lc_median + 5
                lc_std = 5.0
        except:
            lc_median = data_x[p_i] + 25
            lc_p25 = lc_median - 5
            lc_p75 = lc_median + 5
            lc_std = 5.0
        best_score = np.inf
        best_popt = None
        best_pcov = None
        starts = []
        if data_x[p_i] < lc_median < data_x[p_i] + 200:
            starts.append(round(lc_median, 1))
        for val in [lc_p25, lc_p75]:
            v = round(val, 1)
            if data_x[p_i] < val < data_x[p_i] + 200 and v not in starts:
                starts.append(v)
        if not starts:
            starts.append(data_x[p_i] + 25)
        for lc_guess in starts:
            p0 = [lc_guess, 0.36]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt, pcov = curve_fit(lcfunc, x_fit, y_fit, p0=p0,
                        bounds=([data_x[p_i], lp[0]], [data_x[p_i] + 50, lp[1]]))
                y_pred = lcfunc(x_fit, *popt)
                rms = np.sqrt(np.mean((y_fit - y_pred)**2))
                if not (np.isfinite(pcov).all() and np.all(np.diag(pcov) > 0)):
                    pcov = None
                penalty = abs(popt[0] - lc_median) / lc_std
                score = rms + penalty
                if score < best_score:
                    best_score = score
                    best_popt = popt
                    best_pcov = pcov
                if lc_guess == starts[0] and rms < 3.0:
                    break
            except:
                continue
        if best_popt is not None:
            popt = best_popt
            pcov = best_pcov
            y_pred = lcfunc(x_fit, *popt)
            rms = np.sqrt(np.mean((y_fit - y_pred)**2))
            quality = rms <= 8.0
            if pcov is not None:
                lc_err = np.sqrt(abs(pcov[0,0]))
                lp_err = np.sqrt(abs(pcov[1,1]))
            else:
                lc_err = lc_std
                lp_err = 0.05
                quality = False
        else:
            popt = (lc_median, 0.36)
            quality = False
            lc_err = lc_std
            lp_err = 0.05
        x_pre = data_x[b_i:p_i]
        y_pre = data_y[b_i:p_i]
        try:
            _, x_wrc_pre = WRC_transformer(y_pre, x_pre, thr=5)
            lc_wrc_pre = x_wrc_pre[(x_wrc_pre > data_x[p_i]) & (x_wrc_pre < data_x[p_i] + 200)]
            if len(lc_wrc_pre) > 5:
                lc_pre_median = float(np.median(lc_wrc_pre))
                lc_pre_std = max(float(np.std(lc_wrc_pre)), 1.0)
            else:
                lc_pre_median = data_x[p_i] + 25
                lc_pre_std = 5.0
        except:
            lc_pre_median = data_x[p_i] + 25
            lc_pre_std = 5.0
        pre_best_score = np.inf
        pre_best_popt = None
        pre_starts = {lc_pre_median}
        for delta in [-lc_pre_std * 0.5, lc_pre_std * 0.5]:
            v = lc_pre_median + delta
            if data_x[p_i] < v < data_x[p_i] + 50:
                pre_starts.add(round(v, 1))
        for lc_guess_pre in pre_starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt_try, _ = curve_fit(lcfunc, x_pre, y_pre, p0=[lc_guess_pre, 0.36],
                        bounds=([data_x[p_i], 0], [data_x[p_i] + 50, 0.5]))
                y_pred_pre = lcfunc(x_pre, *popt_try)
                rms_pre = np.sqrt(np.mean((y_pre - y_pred_pre)**2))
                penalty_pre = abs(popt_try[0] - lc_pre_median) / lc_pre_std
                score_pre = rms_pre + penalty_pre
                if score_pre < pre_best_score:
                    pre_best_score = score_pre
                    pre_best_popt = popt_try
            except:
                continue
        if pre_best_popt is not None:
            popt_pre = pre_best_popt
        else:
            try:
                popt_pre = (float(np.median(WRC_transformer(y_pre, x_pre, thr=5)[1])), 0.36)
            except:
                popt_pre = (lc_pre_median, 0.36)
        lc, p = popt
        fc.data['wlcarg'].append((lc, p))
        fc.data['slopepre'].append((popt_pre[0],popt_pre[1]))
        fc.data['wlc_quality'].append(quality)
        fc.data['wlc_err'].append((lc_err, lp_err))
    fc.data['wlcarg'] = list(fc.data['wlcarg'])
def peakH(fc):
    peakindex = copy.deepcopy(fc.data['peakindex'])
    fc.data['peakindex']=[]
    data = fc.get_prodata()['retract']
    data_y = data['vDeflection'].reshape(-1)*1e12
    for p_i in peakindex:
        if data_y[p_i]>=fc.data['arg']['peakH']:
            fc.data['peakindex'].append(p_i)
def peakN(fc):
    peakN=fc.data['arg']['peakN']
    peaknum = len(fc.data['peakindex'])
    if peaknum >= peakN[0] and peaknum <= peakN[1]:
        fc.data['peaknum_judge'] = True
    else:
        fc.data['peaknum_judge'] = False
def slope(fc):
    fc.data['k'] = np.array([])
    data = fc.get_prodata()['retract']
    data_x,data_y = data['measuredHeight'].reshape(-1)*1e9,data['vDeflection'].reshape(-1)*1e12
    if fc.data['tasktype']=='smfs':
        for i,arg in enumerate(fc.data['slopepre']):
            k = lcfunc1d(data_x[fc.data['peakindex'][i]],*arg)
            fc.data['k'] = np.append(fc.data['k'],k)
        #del fc.data['slopepre']
    elif fc.data['tasktype']=='cell_curve':
        n = int(0.05*len(data_x))
        for i,p_i in enumerate(fc.data['peakindex']):
            if len(data_x[p_i-n:p_i])>0:
                x,y = data_x[p_i-n:p_i],data_y[p_i-n:p_i]
                k = get_slope(x, y)
            else:
                k = 5
            fc.data['k'] = np.append(fc.data['k'],k)
    fc.data['k'] = list(fc.data['k'])
def countdlc(fc):
    lc = np.array([])
    for arg in fc.data['wlcarg']:
        lc = np.append(lc,arg[0])
    fc.data['dlc'] = np.diff(lc)
def mkbaseondlc(fc):
    dic = fc.data['arg']['mark']
    fc.data['mark'] = []
    dlc = fc.data['dlc']
    peakindex = fc.data['peakindex']
    data_y = None
    has_force = {}
    no_force = {}
    for m, rang in dic.items():
        n = len(rang)
        has_f = (n >= 3 and rang[2] is not None) or (n >= 4 and rang[3] is not None)
        if has_f:
            has_force[m] = rang
        else:
            no_force[m] = rang
    if has_force:
        data = fc.get_prodata()['retract']
        data_y = data['vDeflection'].reshape(-1) * 1e12
    for i in range(len(dlc)):
        matched = False
        if has_force and data_y is not None:
            f = data_y[peakindex[i]]
            for m, rang in has_force.items():
                min_dlc = rang[0]
                max_dlc = rang[1]
                d_ok = dlc[i] > min_dlc and dlc[i] < max_dlc
                if not d_ok:
                    continue
                min_f = rang[2] if len(rang) >= 3 else None
                max_f = rang[3] if len(rang) >= 4 else None
                f_ok = True
                if min_f is not None:
                    f_ok = f_ok and f >= min_f
                if max_f is not None:
                    f_ok = f_ok and f <= max_f
                if d_ok and f_ok:
                    fc.data['mark'].append(m)
                    matched = True
                    break
        if not matched:
            for m, rang in no_force.items():
                min_dlc = rang[0]
                max_dlc = rang[1]
                if dlc[i] > min_dlc and dlc[i] < max_dlc:
                    fc.data['mark'].append(m)
                    matched = True
                    break
        if not matched:
            fc.data['mark'].append('none')
def findpeak_smallrange(data_y,height=10):
    for prominence in range(3,60,2):
        p,h = find_peaks(data_y,height=height,prominence=15,distance=10)
        if len(p)<3:
            break
    return p
def allrun(fc):
    noise_down(fc)
    cal_baseline_drift(fc)
    cal_baseline_y(fc)
    cal_baseline_x(fc)
    predict(fc)
    findpeak(fc)
    wlcfit(fc)
    peakH(fc)
    peakN(fc)
    slope(fc)
    countdlc(fc)
    mkbaseondlc(fc)
def qmWLC_transformer(f,x,thr=30,p=0.36):
    x = x[np.where(f>thr)]*1e-9
    f = f[np.where(f>thr)]*1e-12
    kb = 1.38e-23
    T = 298
    p = p*1e-9
    gama1 = 27.4e-9
    gama2 = 109.8e-9
    ff = f*p/kb/T
    b = np.exp(np.sqrt(900/ff))
    Lc = x/(4/3-4/3/np.sqrt(ff+1)-10*b/np.sqrt(ff)/((b-1)**2)+ff**1.62/(3.55+3.8*ff**2.2))
    L_0 = Lc/(1/2/gama1*np.sqrt(gama1**2+4*gama2*f+2*gama2-gama1))
    state_L = L_0/2/gama2*(np.sqrt(4*f*gama2+gama1**2)-gama1+2*gama2)
    return f*1e12,state_L*1e13
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
def WLC_transformer(f,x,thr=20):
    x = x[np.where(f>thr)].astype(complex)*1e-9
    f = f[np.where(f>thr)].astype(complex)*1e-12
    p = np.array([0.36e-9],dtype=complex)
    lc = wlc2lc(x,f,p)
    return f,lc
def mlti_Gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y
def Lc_transformer(data_x,data_y,plottype='hist'):
    fig,ax = plt.subplots(figsize=(10,6),dpi=300)
    f,x = WRC_transformer(data_y,data_x)
    if plottype=='scatter':
        ax.scatter(x,f,s=2,c='#495057')
        img = fig2img(fig)
        plt.close()
        return img
    a=ax.hist(x,bins=int(x.max()-x.min()))
    kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(x.reshape(-1,1))
    x_ = np.linspace(x.min(),x.max(),int(x.max()-x.min()))
    log_dens = kde.score_samples(x_.reshape(-1,1))
    p,_ = find_peaks(np.exp(log_dens)/np.exp(log_dens).max(),height=0.15,distance=5)
    guess = []
    bound_start = []
    bound_end = []
    for i in p:
        guess += [x_[i], 25, 1]   
        bound_start += [x_[i]-20,0,0]
        bound_end += [x_[i]+20,100,20]
    X = a[1]
    Y = np.append(a[0],0)
    popt, pcov = curve_fit(mlti_Gaussian, X, Y, p0=guess,bounds=(bound_start,bound_end))
    lc = popt[::3]
    for i,l in enumerate(lc):
        if i<len(lc)-1:
            if i%2==0:
                ax.text(l,a[0].max()+10,str(round(lc[i+1]-l,1)),c='b')
            else:
                ax.text(l,a[0].max()+5,str(round(lc[i+1]-l,1)),c='b')
    fit = mlti_Gaussian(x_, *popt)
    ax.plot(x_, fit , 'r')
    ax.set_xlim((lc[0]-30,lc[-1]+50))
    img = fig2img(fig)
    plt.close()
    return img
def Lc_transformer_(x,y):
    f,x = WRC_transformer(y,x)
    a = np.histogram(x,bins=int(x.max()-x.min()))
    kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(x.reshape(-1,1))
    x_ = np.linspace(x.min(),x.max(),int(x.max()-x.min()))
    log_dens = kde.score_samples(x_.reshape(-1,1))
    p,_ = find_peaks(np.exp(log_dens)/np.exp(log_dens).max(),height=0.15,distance=5)
    guess = []
    bound_start = []
    bound_end = []
    for i in p:
        guess += [x_[i], 25, 1]   
        bound_start += [x_[i]-20,0,0]
        bound_end += [x_[i]+20,100,20]
    X = a[1]
    Y = np.append(a[0],0)
    popt, pcov = curve_fit(mlti_Gaussian, X, Y, p0=guess,bounds=(bound_start,bound_end))
    lc = popt[::3]
    return lc
def plotmap(arr):
    lens = int(np.sqrt(len(arr)))
    arr = arr[:lens**2]
    d = arr.reshape((lens,lens))
    d[1::2]=d[1::2][:,::-1]
    fig,ax = plt.subplots(figsize=(8,6),dpi=300)
    plt.axis('off')
    cmap = plt.get_cmap('YlOrBr_r')
    im = ax.pcolormesh(np.arange(lens),np.arange(lens),d,cmap=cmap,shading='auto')
    bar = fig.colorbar(im)
    bar.set_label('Force(pN)')
    img = fig2img(fig)
    plt.close()
    return img
    pass
def plothist(arr):
    fig,ax = plt.subplots(figsize=(8,6),dpi=300)
    ax.hist(arr,bins=50)
    img = fig2img(fig)
    plt.close()
    return img

# ======== extract curve-level features (for classification) ========
def extract_curve_features(fc, mark_defs=None):
    n = len(fc.data.get('peakindex', []))
    marks = fc.data.get('mark', [])
    dlc = fc.data.get('dlc', [])
    wq = fc.data.get('wlc_quality', [])
    peak_forces = fc.data.get('peak_forces', [])
    forces = peak_forces if peak_forces else []
    if mark_defs is None:
        mark_defs = list(fc.data.get('arg', {}).get('mark', {}).keys())
    if not mark_defs:
        mark_defs = ['none']
    features, names = [], []

    for mt in mark_defs:
        indices = [i for i, m in enumerate(marks) if m == mt]
        features.append(len(indices))
        names.append('n_{}'.format(mt.replace(' ', '_')))
        if indices:
            f_vals = [forces[i] for i in indices if i < len(forces)]
            features.append(np.mean(f_vals) if f_vals else 0)
            names.append('F_{}'.format(mt.replace(' ', '_')))
        else:
            features.append(0)
            names.append('F_{}'.format(mt.replace(' ', '_')))

    n_none = sum(1 for m in marks if m == 'none')
    features += [
        n, n_none, n - n_none,
        np.mean(dlc) if len(dlc) > 0 else 0,
        np.std(dlc) if len(dlc) > 1 else 0,
        np.mean(forces) if forces else 0,
        np.std(forces) if len(forces) > 1 else 0,
        np.min(forces) if forces else 0,
        np.max(forces) if forces else 0,
        forces[0] if forces else 0,
        forces[-1] if forces else 0,
        np.mean(wq) if wq else 0,
        fc.data['filters'].get('win_lens', 13),
    ]
    names += ['n_peaks','n_none','n_marked','mean_dLc','std_dLc',
              'mean_F','std_F','min_F','max_F','first_F','last_F',
              'wlc_q_ratio','sg_win']
    return np.array(features), names

# ======== 4. Semi-supervised classifier ========
def train_classifier(zpo, ljp, train_indices=None, progress_cb=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    fc = forcecurve()
    X_all, y_all = [], []
    if train_indices is None:
        train_indices = []
        for i in range(len(zpo)):
            d = zpo[i]
            if d.get('class', 'N') not in ('N', 'Z') and len(d.get('peakindex', [])) > 0:
                train_indices.append(i)
    else:
        train_indices = [i for i in train_indices if zpo[i].get('class', 'N') not in ('N', 'Z') and len(zpo[i].get('peakindex', [])) > 0]
    total = len(train_indices)
    if total < 5:
        return None, None, None, {}, None
    print(f'[Auto-Classify] Training on {total} labeled curves...')
    trained_mark_defs = list(zpo[train_indices[0]].get('arg', {}).get('mark', {}).keys())
    if not trained_mark_defs:
        trained_mark_defs = ['none']
    for j, i in enumerate(train_indices):
        pct = int(10 + 40.0 * j / max(total, 1))
        if progress_cb:
            progress_cb(pct)
        if j % max(1, total // 20) == 0:
            print(f'  [{j}/{total}] Extracting features...')
        d = zpo[i]
        fc.data = d
        if not d.get('peak_forces'):
            fc.recover_force(ljp)
            fc.clean_force()
        fvec, _ = extract_curve_features(fc, mark_defs=trained_mark_defs)
        X_all.append(fvec)
        y_all.append(d.get('class', 'N'))
    if len(X_all) < 5:
        return None, None, None, {}, None
    if progress_cb:
        progress_cb(25)
    X_all = np.nan_to_num(np.vstack(X_all), nan=0, posinf=10000, neginf=-10000)
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)
    if progress_cb:
        progress_cb(28)
    print(f'[Auto-Classify] Fitting SVM on {len(X_all)} samples...')
    clf = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    clf.fit(X_scaled, y_all)
    if progress_cb:
        progress_cb(30)
    from collections import Counter
    return clf, scaler, list(set(y_all)), dict(Counter(y_all)), trained_mark_defs

def predict_curve_class(fc, clf, scaler, mark_defs=None):
    if clf is None:
        return None
    fvec, _ = extract_curve_features(fc, mark_defs=mark_defs)
    fvec = np.nan_to_num(fvec.reshape(1, -1), nan=0, posinf=10000, neginf=-10000)
    X_sc = scaler.transform(fvec)
    pred = clf.predict(X_sc)[0]
    proba = clf.predict_proba(X_sc)[0]
    return pred, dict(zip(clf.classes_, proba))

def learn_class_stats(zpo, ljp):
    from collections import defaultdict
    stats = defaultdict(lambda: {'dlc': [], 'peak_n': [], 'Lc': [], 'Lp': []})
    for i in range(len(zpo)):
        d = zpo[i]
        cls = d.get('class', 'N')
        if cls in ('N', 'Z') or len(d.get('peakindex', [])) == 0:
            continue
        dlc_vals = d.get('dlc', [])
        if len(dlc_vals) > 0:
            stats[cls]['dlc'].extend(dlc_vals.tolist() if hasattr(dlc_vals, 'tolist') else list(dlc_vals))
        stats[cls]['peak_n'].append(len(d.get('peakindex', [])))
        wlc_vals = d.get('wlcarg', [])
        for lc, lp_val in wlc_vals:
            stats[cls]['Lc'].append(lc)
            stats[cls]['Lp'].append(lp_val)
    result = {}
    for cls, vals in stats.items():
        if len(vals['dlc']) < 2:
            continue
        d_median = np.median(vals['dlc'])
        d_std = np.std(vals['dlc'])
        pn_min = int(np.min(vals['peak_n'])) if vals['peak_n'] else 0
        pn_max = int(np.max(vals['peak_n'])) if vals['peak_n'] else 0
        lp_vals_clean = [v for v in vals['Lp'] if 0.1 < v < 1.0]
        lp_median = np.median(lp_vals_clean) if lp_vals_clean else 0.36
        lp_std = np.std(lp_vals_clean) if len(lp_vals_clean) > 1 else 0.05
        result[cls] = {
            'dLc_median': d_median, 'dLc_std': d_std,
            'dLc_min': d_median - 1.5 * d_std, 'dLc_max': d_median + 1.5 * d_std,
            'F_median': 0, 'F_std': 0,
            'peak_n_min': pn_min, 'peak_n_max': pn_max,
            'Lp_median': lp_median, 'Lp_std': lp_std,
        }
    return result

def auto_label_curve(fc, class_stats, predicted_class):
    if predicted_class not in class_stats:
        return
    stat = class_stats[predicted_class]
    dlc = fc.data.get('dlc', [])
    peaks = fc.data.get('peakindex', [])
    if len(dlc) == 0:
        return
    fc.data['mark'] = []
    data = fc.get_prodata()['retract']
    data_y = data['vDeflection'].reshape(-1) * 1e12
    for i in range(len(dlc)):
        f_val = data_y[peaks[i]] if i < len(peaks) else 0
        d_ok = stat['dLc_min'] < dlc[i] < stat['dLc_max']
        f_ok = abs(f_val - stat['F_median']) < 2 * max(stat['F_std'], 10) if stat['F_std'] > 0 else True
        if d_ok and f_ok:
            fc.data['mark'].append(predicted_class)
        else:
            fc.data['mark'].append('none')

def rescue_reanalyze(fc, class_stats, target_class='A'):
    if target_class not in class_stats:
        return False
    if len(fc.data.get('rawdata', {})) == 0:
        return False
    stat = class_stats[target_class]
    saved_lp = fc.data['arg'].get('lp', [0.34, 0.38])
    lp_lo = max(0.20, stat['Lp_median'] - 2 * stat['Lp_std'])
    lp_hi = min(0.60, stat['Lp_median'] + 2 * stat['Lp_std'])
    fc.data['arg']['lp'] = [lp_lo, lp_hi]
    wlcfit(fc)
    countdlc(fc)
    mkbaseondlc(fc)
    fc.data['arg']['lp'] = saved_lp
    return True

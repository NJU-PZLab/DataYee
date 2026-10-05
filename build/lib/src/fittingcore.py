# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:15:07 2021

@author: ZhengBin
"""
import numpy as np
from itertools import product
T = 298
kb = 1.38e-23
gama = 0.577216
import matplotlib.pyplot as plt
from src.fittingEnergy import Ui_fitting
from PyQt5.QtWidgets import QDialog,QMessageBox,QGraphicsScene,QGraphicsPixmapItem,QMenu,QTableView
from src.datapro import is_number,fig2img
from src.ui_related import tableplus
from PyQt5.QtGui import QImage,QPixmap,QCursor
from PyQt5.QtCore import Qt
def BE(x_arr,x_beta,k_off):
    f_beta = kb*T/x_beta
    F_arr = f_beta*np.log(x_arr/f_beta/k_off)
    return F_arr
def Friddle(x_arr,x_beta,k_off,Feq):
    f_beta = kb*T/x_beta
    F_arr = Feq + f_beta*np.log(1+np.e**(-1*gama)*x_arr/(k_off*f_beta))
    return F_arr
def DHS(x_arr,x_beta,k_off,dG):
    pass
def r2_calculate(y_actual: np.ndarray,y_predicted: np.ndarray)->np.float64:
    sse = np.sum((y_actual - y_predicted) ** 2,axis=1)
    sst = np.sum((y_actual - np.mean(y_actual)) ** 2,axis=1)
    r2 = 1 - sse / sst
    return r2
def fit(x_arr,y_arr,bounds,methods='BE',scale_factor=0.3,max_iter=5):
    if methods == 'BE':
        arg_num = 2
        if len(bounds)!=arg_num:
            return False
        func = BE
    elif methods == 'Friddle':
        arg_num = 3
        if len(bounds)==2:
            bounds = np.vstack((bounds,np.array([[0,y_arr.min()]])))
        if len(bounds)!=arg_num:
            return False
        func = Friddle
    elif methods == 'DHS':
        return False
        func = DHS
    else:
        return False
    max_r2 = 0
    best_arg = np.array([])
    for i in range(max_iter):
        b = np.array(list(product(*[np.linspace(x,y) for x,y in bounds]))).T
        arg = [np.tile(x.reshape(-1,1),(1,len(x_arr))) for x in b]
        res = func(x_arr,*arg)
        r2 = r2_calculate(res,y_arr)
        index0_1 = np.where((r2>0)&(r2<1))[0]
        if len(index0_1)==0:
            break
        max_index = index0_1[r2[index0_1].argmax(axis=0)]
        if max_r2 < r2[max_index]:
            max_r2 = r2[max_index]
            max_r2_index = r2.argmax(axis=0)
            best_arg = b.T[max_r2_index]
            bounds = np.tile(best_arg.reshape(-1,1),(1,2))+np.tile(np.diff(bounds)*scale_factor,(1,2))*np.array([-1,1])
            bounds[np.where(bounds<0)]=1e-13
        else:
            break
    if len(best_arg)==0:
        return False
    return {'r_2':max_r2,'arg':best_arg}
def plot(x_arr,y_arr,arg,methods='BE'):
    if methods == 'BE':
        func = BE
    elif methods == 'Friddle':
        func = Friddle
    elif methods == 'DHS':
        return False
        func = DHS
    else:
        return False
    fig,ax = plt.subplots(dpi=100)
    ax.set_xscale('log')
    ax.plot(x_arr*1e12,y_arr*1e12,'ro')
    
    x_ = np.linspace(x_arr.min(),x_arr.max())
    y_ = func(x_,*arg)
    ax.plot(x_*1e12,y_*1e12)
    ax.set_title(methods)
    ax.set_ylabel('Force(pN)')
    ax.set_xlabel('Loading rate(pN/s)')
    return fig

class fitEnergy(QDialog,Ui_fitting,QTableView):
    def __init__(self):
        super(fitEnergy, self).__init__()
        self.setupUi(self)
        self.arglst = ['x_beta','k_off']
        self.argindex = 0
        self.arg = dict(zip(self.arglst,[[0.1,0.9],[0.1,100]]))
        self.pushButton_3.clicked.connect(self.changearg)
        self.doubleSpinBox.valueChanged.connect(self.getarg)
        self.doubleSpinBox_2.valueChanged.connect(self.getarg)
        self.pushButton.clicked.connect(self.calculate)
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)
        self.contextMenu = QMenu(self)
        self.CP = self.contextMenu.addAction('copy')
        self.JQ = self.contextMenu.addAction('cut')
        self.NT = self.contextMenu.addAction('paste')
        self.DL = self.contextMenu.addAction('delete')
        self.tableplus = tableplus(self)
        self.CP.triggered.connect(self.tableplus.copy)
        self.JQ.triggered.connect(self.tableplus.cut)
        self.NT.triggered.connect(self.tableplus.paste)
        self.DL.triggered.connect(self.tableplus.del_tb_text)
        
    def showMenu(self, pos):
        self.contextMenu.exec_(QCursor.pos())
    def start(self,energytype):
        self.energytype = energytype
        self.showlabel()
        self.setspinvalue()
        self.show()
    def setspinvalue(self):
        self.doubleSpinBox.setValue(self.arg[self.arglst[self.argindex]][0])
        self.doubleSpinBox_2.setValue(self.arg[self.arglst[self.argindex]][1])
    def changearg(self):
        if self.argindex<len(self.arglst)-1:
            self.argindex = self.argindex+1
        else:
            self.argindex = 0
        self.showlabel()
        self.setspinvalue()
    def showlabel(self):
        arglabel = self.arglst[self.argindex]
        self.argname.setText(arglabel)
        if arglabel == 'x_beta':
            self.unit.setText('nm')
        else:
            self.unit.setText('')
    def getarg(self,value):
        sender = self.sender()
        if sender == self.doubleSpinBox:
            self.arg[self.arglst[self.argindex]][0] = value
        elif sender == self.doubleSpinBox_2:
            self.arg[self.arglst[self.argindex]][1] = value
    def showimg(self):
        img = self.img
        if img == None:
            self.close()
            return None
        self.img = img
        scale = img.size[0]/589
        #img = img.resize((int(img.size[0]/scale), int(img.size[1]/scale)),Image.ANTIALIAS)
        #img.show()
        self.frame = QImage(np.array(img), img.size[0], img.size[1], QImage.Format_RGB888)
        self.pix = QPixmap.fromImage(self.frame).scaledToWidth(int(img.size[0]/scale)).scaledToHeight(int(img.size[1]/scale))
        self.item = QGraphicsPixmapItem(self.pix)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        self.show()
    def calculate(self):
        bounds = np.array(list(self.arg.values()))
        if len(np.where(np.diff(bounds)<=0)[0])!=0:
            QMessageBox.information(self,"Erroe","Input error!")
            return None
        bounds[0] = bounds[0]*1e-9
        x_arr = np.array([])
        y_arr = np.array([])
        for i in range(1,15):
            y = self.tableWidget.item(i, 0)
            x = self.tableWidget.item(i, 1)
            if None not in [y,x] and is_number(x.text()) and is_number(y.text()):
                y,x = float(y.text()),float(x.text())
                if y<=0 or x<=0:
                    QMessageBox.information(self,"Erroe","Input error!")
                    return None
                x_arr = np.append(x_arr,x)
                y_arr = np.append(y_arr,y)
        x_arr,y_arr = x_arr*1e-12,y_arr*1e-12
        if len(x_arr)<=3:
            QMessageBox.information(self,"Erroe","Too little data!")
            return None
        print(x_arr,x_arr,bounds)
        arg = fit(x_arr,y_arr,bounds,methods=self.energytype)
        if not arg:
            QMessageBox.information(self,"Erroe","Fitting error!")
            return None
        print(arg)
        fig = plot(x_arr, y_arr, arg['arg'],methods=self.energytype)
        fig.savefig('test.png',bbox_inches='tight',transparent=True)
        self.img = fig2img(fig)
        plt.close()
        self.showimg()
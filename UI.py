import sys
import copy
import os
import numpy as np
from src.datapro import lcfunc
from src.loadjpk import forcecurve
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QProgressDialog, QGridLayout, \
    QButtonGroup,QDialog,QGraphicsScene,QGraphicsPixmapItem,QTableWidgetItem,QShortcut, QLabel, QDoubleSpinBox, QPushButton, QCheckBox, QAction

from src.designer import Ui_MainWindow
from src.fittingcore import fitEnergy
from src.ui_related import dlcrange_window,showimage,para_window,statistics_win,script_win
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from src.main import programbody
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mplstyle
from src.multiProcess import multi_run
#mpl.use('agg')
#from PIL import ImageQt,Image

mplstyle.use('fast')
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.subplot.left'] = 0.05
mpl.rcParams['figure.subplot.right'] = 1
mpl.rcParams['figure.subplot.top'] = 1
mpl.rcParams['figure.subplot.bottom'] = 0.05
color_lsts = ['#f76707']*9


# plt.ion()
def getfitcurve(wlcarg, peakindex, data_x):
    arg_lst = []
    for i, arg in enumerate(wlcarg):
        x_ = np.linspace(0, data_x[peakindex[i]] + 10)
        y_ = lcfunc(x_, *arg)
        x_ = x_[:y_.argmax()]
        y_ = y_[:y_.argmax()]
        arg_lst.append((x_, y_))
    return arg_lst

class myFigure(FigureCanvas):
    def __init__(self):
        self.canvas = FigureCanvas(mpl.figure.Figure(dpi=100))
        self.ax = self.canvas.figure.add_subplot()
        self.ax.plot([-1e4, 1e4], [0, 0], lw=1.5, c='#ff8787')
        self.ax.plot([0, 0], [-50, 50], 'r-', lw=1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.5,hspace=0.1,wspace=0.1)
        self.fc = forcecurve()
        self.index = 0
        self.content = {'curve': [],
                        'peak': [],
                        'bottom': [],
                        'mark': [],
                        'fitcurve': [],
                        'k':[],
                        'selrange':[],
                        'class':[]}
        self.range_fix = False
        super(myFigure, self).__init__(self.canvas.figure)
        self.overlay_dic = {}
        self.overlaymode = False
    def zoom_func(self,event,base_scale = 1.1,zoomx_state=True,zoomy_state=True):
        if not zoomx_state and not zoomy_state:
            return None
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata
        if event.button == 'up':
            scale_factor = 1/base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
        if zoomx_state:
            self.ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        if zoomy_state:
            self.ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
    def plot_selrange(self,datax1,datax2):
        if self.fc.data['datamsg'][0]=='':
            return None
        if datax1 ==None or datax2 == None:
            for line in self.content['selrange']:
                try:
                    line[0].remove()
                except:
                    continue
            return None
        if datax1>datax2:
            datax1,datax2=datax2,datax1
        for line in self.content['selrange']:
            try:
                line[0].remove()
            except:
                continue
        cur_ylim = self.ax.get_ylim()
        self.content['selrange'].append(self.ax.plot([datax1,datax1],cur_ylim,c='#9fa8da',lw=1))
        self.content['selrange'].append(self.ax.plot([datax2,datax2],cur_ylim,c='#9fa8da',lw=1))
    def overlay(self,curve_index):
        if 'overlay' not in self.fc.data.keys():
            self.fc.data['overlay'] = self.overlaymode
        index = curve_index
        if self.fc.data['overlay']:
            if index not in self.overlay_dic.keys():
                pass
            else:
                self.overlay_dic[index][0].remove()
                del self.overlay_dic[index]
            self.overlay_dic[index] = self.ax.plot(self.data_x[::10], self.data_y[::10], 'k', lw=1.5,alpha=0.1,markevery=10)
        else:
            if index in self.overlay_dic.keys():
                self.overlay_dic[index][0].remove()
                del self.overlay_dic[index]
    def clean_overlay(self):
        for i,v in self.overlay_dic.items():
            v[0].remove()
        self.overlay_dic = {}
    def motion(self,dx,dy):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        x = (cur_xlim[1] + cur_xlim[0])*0.5-dx
        y = (cur_ylim[1] + cur_ylim[0])*0.5-dy
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        self.ax.set_ylim([y - cur_yrange,y + cur_yrange])
        self.ax.set_xlim([x - cur_xrange,x + cur_xrange])
        
    def getdata(self):
        data = self.fc.get_prodata()['retract']
        self.data_y = data['vDeflection'][:, 0] * 1e12
        self.data_x = data['measuredHeight'][:, 0] * 1e9

    def setlim(self, xlim, ylim):
        if self.range_fix:
            return None
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def plotcurve(self):
        for line in self.content['curve']:
            line[0].remove()
        self.content['curve'] = []
        # self.ax.set_yticks(np.arange(0,self.data_y.max(),150))
        if not self.fc.data['artificial_judge']:
            self.content['curve'].append(self.ax.plot(self.data_x, self.data_y, 'b', lw=1.5))
        else:
            self.content['curve'].append(self.ax.plot(self.data_x, self.data_y, c='#495057', lw=1.5))

    def plotfitcurve(self):
        for line in self.content['fitcurve']:
            line[0].remove()
        self.content['fitcurve'] = []
        if len(self.fc.data['peakindex']) <= 0:
            return None
        data_x = self.fc.get_prodata()['retract']['measuredHeight'][:, 0] * 1e9
        fit_lst = getfitcurve(self.fc.data['wlcarg'], self.fc.data['peakindex'], data_x)
        color_lst = (len(self.fc.data['wlcarg']) // len(color_lsts) + 1) * color_lsts
        for i, xy_ in enumerate(fit_lst):
            x_, y_ = xy_
            self.content['fitcurve'].append(self.ax.plot(x_, y_, '-.', c=color_lst[i], lw=1.5))

    def plotpeak(self):
        for line in self.content['peak']:
            line[0].remove()
        self.content['peak'] = []
        peak_index = self.fc.data['peakindex']
        if len(peak_index) <= 0:
            return None
        
        
        self.content['peak'].append(
            (self.ax.plot(self.data_x[peak_index], self.data_y[peak_index], 'yo', markersize=8)))
        self.content['peak'].append(
            self.ax.plot(self.data_x[peak_index[self.index]], self.data_y[peak_index[self.index]],'ro', markersize=8))
        '''
    def plotbott8om(self):
        for line in self.content['bottom']:
            line[0].remove()
        self.content['bottom'] = []
        bottom_index = self.fc_new.data['bottomindex']
        if len(bottom_index) <= 0:
            return None
        self.content['bottom'].append(
            self.ax.plot(self.data_x[bottom_index], self.data_y[bottom_index], 'g*', markersize=8))
        '''
    def plotmark(self):
        for line in self.content['mark']:
            line.remove()
        self.content['mark'] = []
        font = {'family': 'serif', 'style': 'italic', 'weight': 'normal', 'color': 'red', 'size': 14}
        mark = self.fc.data['mark']
        if len(mark) <= 0:
            return None
        peak_index = self.fc.data['peakindex']
        for i in range(len(mark)):
            if mark[i] == 'none':
                font['color'] = 'red'
            else:
                font['color'] = 'blue'
            if i % 3 == 0:
                self.content['mark'].append(
                    self.ax.text(self.data_x[peak_index[i]] - 3, -70, '{}.{}'.format(i, mark[i]), font,
                                 horizontalalignment='left'))
            elif i % 3 == 1:
                self.content['mark'].append(
                    self.ax.text(self.data_x[peak_index[i]] - 3, -50, '{}.{}'.format(i, mark[i]), font,
                                 horizontalalignment='left'))
            elif i % 3 == 2:
                self.content['mark'].append(
                    self.ax.text(self.data_x[peak_index[i]] - 3, -30, '{}.{}'.format(i, mark[i]), font,
                                 horizontalalignment='left'))
    def plotk(self):
        for line in self.content['k']:
            line[0].remove()
        self.content['k'] = []
        peak_index = self.fc.data['peakindex']
        if len(peak_index)==0:
            return None
        k_lst = self.fc.data['k']
        xrange = (self.data_x[peak_index[-1]]-self.data_x[0])*0.04
        yrange = (self.data_y.max()-self.data_y.min())*0.08
        for i,p_i in enumerate(peak_index):
            x,y = self.data_x[p_i],self.data_y[p_i]
            b = y - k_lst[i]*x
            x_ = np.linspace(x-xrange,x+xrange)
            y_ = k_lst[i]*x_+b
            xyrange_index = np.where((y_<y+yrange)&(y_>y-yrange))[0]
            x_ = x_[xyrange_index]
            y_ = y_[xyrange_index]
            self.content['k'].append(self.ax.plot(x_,y_,'b',lw=1))
    def plotclass(self):
        for line in self.content['class']:
            line.remove()
        self.content['class'] = []
        if 'class' not in self.fc.data.keys():
            c = 'N'
        else:
            c = self.fc.data['class']
        self.content['class'].append(self.ax.text(0.05,0.9,c,fontsize=20,fontweight='bold',horizontalalignment='center',verticalalignment='center',transform = self.ax.transAxes))
    def changeall(self):
        self.plotcurve()
        dx = np.abs(self.data_x.max())-self.data_x.min()
        dy = np.abs(self.data_y.max())-self.data_y.min()
        if len(self.fc.data['peakindex']) > 0:
            self.setlim((-10, self.data_x[self.fc.data['peakindex'][-1]] + 0.05*dx), (-90, self.data_y.max() + 0.1*dy))
        else:
            self.setlim((-10, self.data_x.max() + 30), (-90, self.data_y.max() + 40))
        self.plotfitcurve()
        self.plotpeak()
        #self.plotbottom()
        self.plotmark()
        self.plotk()
        self.plotclass()

    def plot(self, fc, index, ljp, tasktype='smfs',curve_index=0):
        self.fc = fc
        self.fc.recover_force(ljp)
        self.index = index
        self.getdata()
        self.changeall()
        self.overlay(curve_index)
        if tasktype == 'cell_curve' and not self.range_fix:
            set_range = 0.1
            ylim_min = self.data_y[int(set_range*len(self.data_y)):].min()-20
            self.setlim((self.data_x.min()-20, self.data_x.max() + 0.1*(self.data_x.max()-self.data_x.min())), (ylim_min, self.data_y.max() + 10))
        plt.draw()



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, m_run,parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self._install_exception_hook()
        self.filedir = ''
        self.fname = ''
        self.tasktype = 'smfs'
        self.svfname = 'test.DataYee-force'
        self.pb = programbody()
        self.F = myFigure()
        self.gridlayout = QGridLayout(self.groupBox)
        self.gridlayout.addWidget(self.F.canvas)
        self.xdata,self.xdata_r = None,None
        self.ydata,self.ydata_r = None,None
        self.zoomx_state =True
        self.zoomy_state = True
        self.zoomfix_state = False
        self.press=False
        self.control,self.alt = False,False
        self.img = None
        self.allowrotate = False
        self.overlaymode = False
        self.showimage_win = showimage()
        self.fitEnergy = fitEnergy()
        self.action_init()
        self.m_run = m_run

    def _install_exception_hook(self):
        import traceback as _tb
        def _hook(exc_type, exc_value, exc_tb):
            try:
                msg = str(exc_value)
            except:
                msg = 'Unknown error'
            self.statusBar().showMessage(f'Error: {msg}', 10000)
            _tb.print_exception(exc_type, exc_value, exc_tb)
        sys.excepthook = _hook

    def action_init(self):
        self.F.canvas.mpl_connect("button_press_event", self.on_press)
        self.F.canvas.mpl_connect('scroll_event',self.scroll_event)
        self.F.canvas.mpl_connect('motion_notify_event',self.onmotion_event)
        self.F.canvas.mpl_connect("button_release_event", self.on_release)
        #self.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.siglestep = 5
        self.lc_value = self.lcdoubleSpinBox.value()
        self.lp_value = self.lpdoubleSpinBox.value()
        self.k_value = self.kSpinBox.value()
        self.sc_default = None
        self.groupBox_11.setMaximumSize(QtCore.QSize(16777215, 210))
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout.setSpacing(5)
        self.lcslide.setGeometry(QtCore.QRect(130, 44, 160, 22))
        self.lpslide.setGeometry(QtCore.QRect(130, 84, 160, 22))
        self.kSlider.setGeometry(QtCore.QRect(130, 124, 160, 22))
        self.comboBox.setGeometry(QtCore.QRect(300, 80, 91, 31))
        self.scLabel = QLabel('SC', self.groupBox_6)
        self.scLabel.setGeometry(QtCore.QRect(10, 160, 41, 31))
        self.scSpinBox = QDoubleSpinBox(self.groupBox_6)
        self.scSpinBox.setGeometry(QtCore.QRect(50, 160, 71, 31))
        self.scSpinBox.setDecimals(4)
        self.scSpinBox.setRange(0.0001, 100.0)
        self.scSpinBox.setSingleStep(0.01)
        self.scSpinBox.setProperty("value", 0.01)
        self.scSetButton = QPushButton('Set', self.groupBox_6)
        self.scSetButton.setGeometry(QtCore.QRect(125, 160, 42, 31))
        self.scSetButton.clicked.connect(self.sc_set)
        self.scClearButton = QPushButton('Clr', self.groupBox_6)
        self.scClearButton.setGeometry(QtCore.QRect(177, 160, 42, 31))
        self.scClearButton.clicked.connect(self.sc_clear)
        self.scSetAllButton = QPushButton('Set All', self.groupBox_6)
        self.scSetAllButton.setGeometry(QtCore.QRect(229, 160, 60, 31))
        self.scSetAllButton.clicked.connect(self.sc_set_all)
        self.actionForce_Curve.triggered.connect(self.openfile)
        self.actionSave.triggered.connect(self.savefile)
        self.actionSave_as.triggered.connect(self.saveasfile)
        self.actionBatch_of_Force_Curve.triggered.connect(self.opendir)
        self.actionDataYee_Force.triggered.connect(self.openfile_DataYee)
        self.actionDataYee.triggered.connect(self.aboutprogramm)
        self.pushButton.clicked.connect(self.run)
        self.downButton.clicked.connect(self.indexplus)
        self.topButton.clicked.connect(self.indexreduct)
        self.leftButton.clicked.connect(self.peakindexretact)
        self.rightButton.clicked.connect(self.peakindexplus)
        self.trashButtom.clicked.connect(self.peakdelete)
        self.actionDelete.triggered.connect(self.forcedelete)
        self.pushButton_2.clicked.connect(self.reset_delete)
        self.lcslide.valueChanged[int].connect(self.lcslidechange)
        self.lpslide.valueChanged[int].connect(self.lpslidechange)
        self.kSlider.valueChanged[int].connect(self.kslidechange)
        self.lcdoubleSpinBox.valueChanged.connect(self.spinbox_changevalue)
        self.lpdoubleSpinBox.valueChanged.connect(self.spinbox_changevalue)
        self.kSpinBox.valueChanged.connect(self.spinbox_changevalue)
        self.spinBox_2.valueChanged.connect(self.spinbox_changevalue)
        self.spinBox_3.valueChanged.connect(self.spinbox_changevalue)
        self.spinBox.valueChanged.connect(self.spinbox_changevalue)
        self.actionexcel.triggered.connect(self.exportexcel)
        self.actionBaseline_plus.triggered.connect(self.baselineplus)
        self.actionBaseline_minus.triggered.connect(self.baselineminus)
        self.actionHistgram.triggered.connect(self.plot_contourhist)
        self.actionScatter.triggered.connect(self.plot_contourscatter)
        self.actionMap.triggered.connect(self.adhesionmap)
        self.actionHistogram.triggered.connect(self.adhesionhist)
        self.actiontxt.triggered.connect(self.exporttxt)
        self.actionbatch_of_txt.triggered.connect(self.exportbatchtxt)
        self.actionpeakindex_plus.triggered.connect(self.peakvalueplus)
        self.actionpeakindex_minus.triggered.connect(self.peakvalueminus)
        self.actioncopy_peak.triggered.connect(self.copypeak)
        self.actionfigure.triggered.connect(self.export_figure)
        self.setFocusPolicy(Qt.StrongFocus)
        self.bg = QButtonGroup(self)
        self.bg.addButton(self.radioButton_2, 0)
        self.bg.addButton(self.tasktype_cell, 1)
        self.bg.buttonClicked.connect(self.rbclicked)
        self.usemodel_cb.setChecked(True)
        self.usemodel_cb.stateChanged.connect(self.statemodel)
        self.stickmodel.setChecked(False)
        self.stickmodel.stateChanged.connect(self.statemodel)
        self.fastmode.setChecked(False)
        self.fastmode.stateChanged.connect(self.statemodel)
        self.groupBox_9.setGeometry(QtCore.QRect(10, 10, 201, 115))
        self.filterdlc_cb = QCheckBox('dLc Filter', self.groupBox_9)
        self.filterdlc_cb.setGeometry(QtCore.QRect(10, 85, 180, 25))
        self.filterdlc_cb.setChecked(True)
        self.filterdlc_cb.stateChanged.connect(self.statemodel)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 135, 201, 111))
        self.groupBox_3.setGeometry(QtCore.QRect(10, 256, 191, 101))
        self.pushButton.setGeometry(QtCore.QRect(0, 367, 231, 51))
        self.lineEdit.returnPressed.connect(self.changemark)
        self.zoomx.setChecked(self.zoomx_state)
        self.zoomy.setChecked(self.zoomy_state)
        self.zoomfix.setChecked(self.zoomfix_state)
        self.zoomx.stateChanged.connect(self.choose_zoom)
        self.zoomy.stateChanged.connect(self.choose_zoom)
        self.zoomfix.stateChanged.connect(self.choose_zoom)
        self.comboBox.currentIndexChanged[str].connect(self.comboBoxchange)
        self.actionBell_Evans.triggered.connect(self.enerpytypeBE)
        self.actionFriddle.triggered.connect(self.enerpytypeF)
        self.actionClustering_by_KMeans.triggered.connect(self.KNcluster)
        self.actionSort_by_similarity.triggered.connect(self.SimilaritySort)
        self.actionAutoClassify = QAction('Auto-Classify', self)
        self.actionAutoClassify.triggered.connect(self.run_auto_classify)
        self.menuClustering.addSeparator()
        self.menuClustering.addAction(self.actionAutoClassify)
        self.actionExit.triggered.connect(self.close)
        self.actionData_equipment.triggered.connect(self.dataEuipment)
        self.actionData_slimming.triggered.connect(self.dataSlimming)
        self.undo_shortcut = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.undo_shortcut.activated.connect(self.undo_action)
    def enerpytypeBE(self):
        self.fitEnergy.start('BE')
    def enerpytypeF(self):
        self.fitEnergy.start('Friddle')
    def onmotion_event(self,event):
        if self.press and None not in [self.ydata,self.xdata,event.xdata,event.ydata]:
            dx = event.xdata-self.xdata
            dy = event.ydata-self.ydata
            if self.control:
                self.F.plot_selrange(self.xdata, event.xdata)
                self.displace_result(range_fix=True)
            elif self.alt:
                self.pb.offset_move(dx,dy,True)
            else:
                self.F.motion(dx, dy)
            self.displace_result(range_fix=True)
            
                
            
    def on_release(self,event):
        self.press=False
        self.xdata_r,self.ydata_r = event.xdata,event.ydata
        if None not in[self.xdata_r,self.xdata]:
            dx = event.xdata-self.xdata
            dy = event.ydata-self.ydata
            if self.control:
                self.pb.rebaseline_cal(self.xdata,event.xdata,allowRotate = self.allowrotate)
                self.F.plot_selrange(None, None)
                self.displace_result(range_fix=True)
            elif self.alt:
                self.pb.offset_move(dx,dy,False)
                
    def choose_zoom(self):
        if self.zoomy.isChecked():
            self.zoomy_state = True
        else:
            self.zoomy_state = False
        if self.zoomx.isChecked():
            self.zoomx_state = True
        else:
            self.zoomx_state = False  
        if self.zoomfix.isChecked():
            self.zoomfix_state = True
        else:
            self.zoomfix_state = False 
    def on_press(self, event):
        self.press=True
        if event.xdata==None or event.ydata==None:
            return None
        self.xdata = event.xdata
        self.ydata = event.ydata
        self.pb.coor_data = (self.xdata,self.ydata)
        self.pb.pk_indexchange(n=None,coor=(self.xdata,self.ydata))
        self.displace_result(range_fix=True)
        
    def scroll_event(self,event):
        if self.xdata!=None and self.ydata!=None:
            event.xdata,event.ydata = self.xdata,self.ydata
        self.F.zoom_func(event,zoomx_state=self.zoomx_state,zoomy_state=self.zoomy_state)
        self.displace_result(range_fix=True)

    def openfile(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load force curve", '*.txt;;*.jpk-force;;*.jpk-force-map;;*.spm')
        self.fname = fname
        if fname != '':
            self.pb.creattask(fname)

    def openfile_DataYee(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open DataYee Force", '*.DataYee-force')
        self.fname = fname
        if fname != '':
            self.pb.creattask(fname)
            self.F.clean_overlay()
            self.displace_result()

    def opendir(self):
        path = QFileDialog.getExistingDirectory(self, 'Load batch of force curve', '*.*')
        self.filedir = path
        if path != '':
            self.pb.creattask(path)

    def savefile(self):
        self.svfname = self.pb.zpo.fname
        if self.svfname == 'test.DataYee-force':
            svfname, _ = QFileDialog.getSaveFileName(self, 'Save DataYee Force', '*.DataYee-force')
            if svfname == '':
                return
            self.svfname = svfname
        self.pb.savechange(self.svfname,saveas=True)
    def saveasfile(self):
        self.svfname = self.pb.zpo.fname
        svfname, _ = QFileDialog.getSaveFileName(self, 'Save DataYee Force', '*.DataYee-force')
        if svfname == '':
            QMessageBox.information(self,"Error","Empty Path!")
        else:
            self.svfname = svfname
            self.pb.savechange(self.svfname,saveas=True)

    def aboutprogramm(self):
        _ = QMessageBox.information(self, 'DataYee', 'Programm Version:0.1', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    def displace_result(self,range_fix=False):
        try:
            self._displace_result_impl(range_fix)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f'Error: {e}', 8000)

    def _displace_result_impl(self,range_fix=False):
        if range_fix:
            self.F.range_fix = True
        else:
            self.F.range_fix = False
        self.gridlayout.removeWidget(self.F.canvas)
        try:
            self.pb.plot(self.F)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f'Plot error: {e}', 8000)
            self.gridlayout.addWidget(self.F.canvas)
            return
        try:
            self.pb.drawlabel(self.label, self.lclplabel)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f'Label error: {e}', 8000)
        self.F.canvas.draw()
        self.gridlayout.addWidget(self.F.canvas)
        if self.pb.state:
            override = self.pb.fc.data.get('springConstant_override')
            if override is not None:
                self.scSpinBox.setValue(override)
            elif self.sc_default is not None:
                self.scSpinBox.setValue(self.sc_default)
            else:
                sc = self.pb.fc.data.get('springConstant', 0.01)
                self.scSpinBox.setValue(sc)
        

    def indexplus(self):
        self.pb.fc_indexchange(1)
        self.resetslide()
        self.spinBox.setValue(self.pb.forcecurve_index)
        self.displace_result(range_fix=self.zoomfix_state)

    def indexreduct(self):
        self.pb.fc_indexchange(-1)
        self.resetslide()
        self.spinBox.setValue(self.pb.forcecurve_index)
        self.displace_result(range_fix=self.zoomfix_state)

    def peakindexplus(self):
        self.pb.pk_indexchange(1)
        self.resetslide()
        self.displace_result(range_fix=self.zoomfix_state)
    def peakvalueplus(self):
        self.pb.pv_change(self.siglestep)
        self.displace_result(range_fix=self.zoomfix_state)
    def peakvalueminus(self):
        self.pb.pv_change(self.siglestep*-1)
        self.displace_result(range_fix=self.zoomfix_state)

    def peakindexretact(self):
        self.pb.pk_indexchange(-1)
        self.resetslide()
        self.displace_result(range_fix=self.zoomfix_state)

    def peakdelete(self):
        self.pb.pk_delete()
        self.resetslide()
        self.displace_result(range_fix=self.zoomfix_state)

    def forcedelete(self):
        self.pb.fc_delete()
        self.indexplus()
        self.resetslide()
        self.displace_result(range_fix=self.zoomfix_state)

    def reset_delete(self):
        self.pb.reset()
        self.displace_result(range_fix=self.zoomfix_state)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up:
            self.indexreduct()
        elif e.key() == Qt.Key_Down:
            self.indexplus()
        elif e.key() == Qt.Key_Left:
            self.peakindexretact()
        elif e.key() == Qt.Key_Right:
            self.peakindexplus()
        elif e.key() == Qt.Key_Delete:
            self.peakdelete()
        elif e.key()==Qt.Key_Control:
            self.control = True
        elif e.key()==Qt.Key_Alt:
            self.alt = True
        elif e.key()>=65 and e.key()<=90:
            self.pb.changeClass(chr(e.key()))
            self.indexplus()
            self.displace_result(range_fix=self.zoomfix_state)
        elif e.key()==Qt.Key_1:
            #self.forcedelete()
            self.CurveKeep()
        elif e.key()==Qt.Key_2:
            self.CurveDiscard()
    def keyReleaseEvent(self,e):
        if e.key()==Qt.Key_Control:
            self.control = False
        elif e.key()==Qt.Key_Alt:
            self.alt = False
    def resetslidevalue(self):
        self.lcslide.setValue(0)
        self.lpslide.setValue(0)
    def changemark(self):
        mark = self.lineEdit.text()
        if mark!='':
            self.pb.changemark(mark)
            self.displace_result(range_fix=self.zoomfix_state)
    def lcslidechange(self, value):
        self.lc_value = self.lcdoubleSpinBox.value()
        self.lp_value = self.lpdoubleSpinBox.value()
        self.pb.lc_change(value, self.lc_value)
        self.displace_result(range_fix=self.zoomfix_state)

    def lpslidechange(self, value):
        self.lc_value = self.lcdoubleSpinBox.value()
        self.lp_value = self.lpdoubleSpinBox.value()
        self.pb.lp_change(value, self.lp_value)
        self.displace_result(range_fix=self.zoomfix_state)
    def kslidechange(self,value):
        self.k_value = self.kSpinBox.value()
        self.pb.k_change(value, self.k_value)
        self.displace_result(range_fix=self.zoomfix_state)

    def sc_set(self):
        value = self.scSpinBox.value()
        self.sc_default = value
        self.pb.set_springconstant_override(value)
        self.displace_result(range_fix=self.zoomfix_state)

    def sc_clear(self):
        self.sc_default = None
        self.pb.clear_springconstant_override()
        self.displace_result(range_fix=self.zoomfix_state)

    def sc_set_all(self):
        value = self.scSpinBox.value()
        reply = QMessageBox.question(self, 'Set All SC',
            f'Set spring constant {value} N/m for all {len(self.pb.zpo)} curves?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.sc_default = value
        self.pb.set_springconstant_override_all(value)
        self.displace_result(range_fix=self.zoomfix_state)

    def resetslide(self):
        self.lcslide.setValue(0)
        self.lpslide.setValue(0)
        self.kSlider.setValue(0)
    def plot_contourhist(self):
        self.img = self.pb.plot_contourhist()
        self.showimage_win.show_img(self.img)
    def plot_contourscatter(self):
        self.img = self.pb.plot_contourscatter()
        self.showimage_win.show_img(self.img)
    def adhesionmap(self):
        self.img = self.pb.adhesionmap()
        self.showimage_win.show_img(self.img)
    def adhesionhist(self):
        self.img = self.pb.adhesionhist()
        self.showimage_win.show_img(self.img)

    def spinbox_changevalue(self, value):
        sender = self.sender()
        if sender == self.spinBox_2:
            self.pb.taskarg['peakN'][0] = value
        elif sender == self.spinBox_3:
            self.pb.taskarg['peakN'][1] = value
        if not self.pb.state:
            return None
        if sender == self.lcdoubleSpinBox:
            self.lc_value = value
        elif sender == self.lpdoubleSpinBox:
            self.lp_value = value
        elif sender == self.kSpinBox:
            self.k_value = value
        elif sender == self.spinBox:
            if value < len(self.pb.zpo):
                self.pb.forcecurve_index = value
            else:
                self.pb.forcecurve_index = len(self.pb.zpo) - 1
            self.displace_result(range_fix=self.zoomfix_state)
        
    def comboBoxchange(self,value):
        if value == 'Fix lc':
            self.pb.fixlc_changelp = True
        else:
            self.pb.fixlc_changelp = False

    def rbclicked(self):
        sender = self.sender()
        if sender == self.bg:
            if self.bg.checkedId() == 0:
                self.tasktype = 'smfs'
            elif self.bg.checkedId() == 1:
                self.tasktype = 'cell_curve'
    def copypeak(self):
        if self.xdata==None:
            return None
        self.pb.copypeak(self.xdata,self.ydata)
        self.displace_result(range_fix=self.zoomfix_state)
    def statemodel(self):
        self.pb.taskarg['usemodel'] = self.usemodel_cb.isChecked()
        self.pb.taskarg['modelstrict'] = self.stickmodel.isChecked()
        self.pb.taskarg['filter_dlc'] = self.filterdlc_cb.isChecked()
        if self.fastmode.isChecked():
            self.pb.taskarg['fastmode'] = 'series'
        else:
            self.pb.taskarg['fastmode'] = 'img'

    def baselineplus(self):
        self.pb.baseline_change(5e-12)
        self.displace_result(range_fix=self.zoomfix_state)

    def baselineminus(self):
        self.pb.baseline_change(-5e-12)
        self.displace_result(range_fix=self.zoomfix_state)

    def exportexcel(self):
        self.pb.export_prodata(self)
    def exporttxt(self):
        self.pb.exporttxt()
    def exportbatchtxt(self):
        self.pb.exportbatchtxt()
    def export_figure(self):
        self.pb.export_figure(self.F.figure)
    def SimilaritySort(self):
        self.pb.SimilaritySort(m_run)
    def KNcluster(self):
        self.pb.KNcluster(m_run)
    def run_auto_classify(self):
        try:
            progress = QProgressDialog(self)
            progress.setWindowTitle('Auto-Classify')
            progress.setLabelText('Training SVM...')
            progress.setCancelButtonText('Cancel')
            progress.setRange(0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            result = self.pb.train_and_auto_classify(progress, self)
        except Exception as e:
            self.statusBar().showMessage(f'Auto-classify error: {e}', 8000)
            return
        if result is not None:
            self.statusBar().showMessage(result, 8000)
            self.displace_result()
    def run(self):
        if not self.pb.ready_run:
            return None
        progress = QProgressDialog(self)
        self.pb.tasktype = self.tasktype
        self.pb.execu_autostep(progress, self)
        if self.pb.state:
            self.displace_result()
    def dataEuipment(self):
        self.pb.Data_equip_slim('e',QProgressDialog(self),self)
    def dataSlimming(self):
        self.pb.Data_equip_slim('s',QProgressDialog(self),self)
    def CurveKeep(self):
        self.pb.keepcurve()
        self.displace_result(range_fix=self.zoomfix_state)
    def CurveDiscard(self):
        self.pb.discardcurve()
        self.displace_result(range_fix=self.zoomfix_state)

    def undo_action(self):
        if self.pb.undo():
            self.resetslide()
            self.spinBox.setValue(self.pb.forcecurve_index)
            self.displace_result(range_fix=self.zoomfix_state)

    def closeEvent(self, event):
        if not self.pb.dirty:
            event.accept()
            return
        reply = QMessageBox.question(self, 'DataYee',
            'There are unsaved changes. Do you want to save before exiting?',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if reply == QMessageBox.Save:
            self.savefile()
            if self.pb.dirty:
                event.ignore()
                return
            event.accept()
        elif reply == QMessageBox.Discard:
            event.accept()
        else:
            event.ignore()
def win_connect(main_win,dialog):
    main_win.actionparameters_setting.triggered.connect(dialog[0].show)
    main_win.actionmark_base_on_dlc.triggered.connect(dialog[1].show)
    main_win.actionHistogram_scatter.triggered.connect(dialog[2].show)
    main_win.actionScript.triggered.connect(dialog[3].show)
    
    
def main():
    app = QApplication(sys.argv)
    m_run = multi_run('None')
    myWin = MyMainWindow(m_run)
    child_window0 = para_window(myWin.pb,myWin)
    child_window1 = dlcrange_window(myWin)
    child_window2 = statistics_win(myWin)
    child_window3 = script_win(myWin)
    win_connect(myWin,[child_window0,child_window1,child_window2,child_window3])
    myWin.show()
    try:
        return app.exec_()
    finally:
        plt.close()


if __name__ == '__main__':
    sys.exit(main())

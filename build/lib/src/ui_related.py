# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 21:40:37 2021

@author: ZhengBin
"""
from PyQt5 import QtCore
from PyQt5.QtGui import QImage,QPixmap,QFont
from PyQt5.QtWidgets import QDialog,QMessageBox,QGraphicsScene,QGraphicsPixmapItem,QApplication,QTableWidgetItem,QGridLayout,QFileDialog,QAbstractItemView,QLabel,QSpinBox
from src.parameters import Ui_Dialog
from src.dlcrange import Ui_dlc_range
from src.showimage import Ui_image
from src.datapro import is_number
from src.scatter_histogramm import Ui_hist_scatter 
from src.script import Ui_Script
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
import matplotlib.pyplot as plt
from importlib import reload
class tableplus():
    def __init__(self,table):
        self.table = table
    def del_tb_text(self):
        try:
            selected_ranges = self.table.tableWidget.selectedRanges()[0]
            for row in range(selected_ranges.topRow(), selected_ranges.bottomRow() + 1):
                for col in range(selected_ranges.leftColumn(), selected_ranges.rightColumn() + 1):
                    newItem = QTableWidgetItem()
                    self.table.tableWidget.setItem(row, col, newItem)
        except BaseException as e:
            print(e)
            return
    
    def paste_tb_text(self):
        try:
            text = QApplication.clipboard().text()
            lst = text.split('\n')[:-1]
            lst1 = []
            for row in lst:
                lst1.append(row.split('\t'))
            selected_ranges = self.table.tableWidget.selectedRanges()[0]
            for r_i,row in enumerate(range(selected_ranges.topRow(), selected_ranges.topRow()+len(lst))):
                for c_i,col in enumerate(range(selected_ranges.leftColumn(), selected_ranges.leftColumn()+len(lst1[0]))):
                    newItem = QTableWidgetItem(lst1[r_i][c_i])
                    self.table.tableWidget.setItem(row, col, newItem)
        except Exception as e:
            print(e)
            return None
    
    def selected_tb_text(self):
        try:
            text_str = ''
            selected_ranges = self.table.tableWidget.selectedRanges()[0]
            for row in range(selected_ranges.topRow(), selected_ranges.bottomRow()+1):
                row_str = ""
                for col in range(selected_ranges.leftColumn(), selected_ranges.rightColumn()+1):
                    item = self.table.tableWidget.item(row, col)
                    if item == None:
                        row_str += ' ' + '\t'
                    else:
                        row_str += item.text() + '\t'
                text_str += row_str + '\n'
            clipboard = QApplication.clipboard() 
            clipboard.setText(text_str)
            return text_str
        except BaseException as e:
            print(e)
            return None
 
    def copy(self):
        text = self.selected_tb_text()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
 
    def cut(self):
        self.copy()
        self.del_tb_text()
 
    def paste(self):
        self.paste_tb_text()
class para_window(QDialog,Ui_Dialog):
    def __init__(self,pb,myWin):
        super(para_window, self).__init__()
        self.setupUi(self)
        self.pb=pb
        self.myWin = myWin
        self.myWin.allowrotate = False
        self.action_init()
    def action_init(self):
        self.sens.valueChanged.connect(self.spinbox_changevalue)
        self.xlimit.valueChanged.connect(self.spinbox_changevalue)
        self.xsens.valueChanged.connect(self.spinbox_changevalue)
        self.peakH.valueChanged.connect(self.spinbox_changevalue)
        self.highspeed.toggled.connect(self.hispeedcorrect)
        self.spinBox.valueChanged.connect(self.spinbox_changevalue)
        self.lp_min.valueChanged.connect(self.spinbox_changevalue)
        self.lp_max.valueChanged.connect(self.spinbox_changevalue)
        self.rotatestate.setChecked(False)
        self.rotatestate.stateChanged.connect(self.cbchange)
        self.delay.stateChanged.connect(self.cbchange)
        self.includeLast.stateChanged.connect(self.cbchange)
        self.resize(617, 440)
        self.sgLabel = QLabel('SG window', self)
        self.sgLabel.setGeometry(QtCore.QRect(40, 330, 171, 31))
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.sgLabel.setFont(font)
        self.sgWin = QSpinBox(self)
        self.sgWin.setGeometry(QtCore.QRect(220, 330, 91, 41))
        font_s = QFont()
        font_s.setFamily("Arial")
        font_s.setPointSize(12)
        self.sgWin.setFont(font_s)
        self.sgWin.setRange(0, 51)
        self.sgWin.setSingleStep(2)
        self.sgWin.setProperty("value", 0)
        self.sgWin.valueChanged.connect(self.spinbox_changevalue)
        self.sgLabel2 = QLabel('auto=0', self)
        self.sgLabel2.setGeometry(QtCore.QRect(320, 340, 72, 15))
        font2 = QFont()
        font2.setFamily("Arial")
        font2.setPointSize(12)
        self.sgLabel2.setFont(font2)
        self.buttonBox.setGeometry(QtCore.QRect(260, 385, 341, 32))
        self.buttonBox.accepted.connect(self.apply_parameters)
        self.load_parameters()

    def showEvent(self, event):
        self.load_parameters()
        super().showEvent(event)

    def load_parameters(self):
        arg = self.pb.taskarg
        widgets = [self.sens, self.xlimit, self.xsens, self.peakH,
                   self.lp_min, self.lp_max, self.sgWin, self.highspeed,
                   self.includeLast]
        for widget in widgets:
            widget.blockSignals(True)
        self.sens.setValue(arg.get('sens', 10))
        self.xlimit.setValue(arg.get('xlim', 20))
        self.xsens.setValue(arg.get('xsens', 2))
        self.peakH.setValue(arg.get('peakH', 30))
        lp = arg.get('lp', [0.34, 0.38])
        self.lp_min.setValue(lp[0])
        self.lp_max.setValue(lp[1])
        self.sgWin.setValue(arg.get('sg_win_lens', 0))
        self.highspeed.setChecked(arg.get('highspeed', False))
        self.includeLast.setChecked(self.pb.taskarg.get('include_last', True))
        for widget in widgets:
            widget.blockSignals(False)

    def apply_parameters(self):
        self.pb.taskarg['sens'] = self.sens.value()
        self.pb.taskarg['xlim'] = self.xlimit.value()
        self.pb.taskarg['xsens'] = self.xsens.value()
        self.pb.taskarg['peakH'] = self.peakH.value()
        self.pb.taskarg['lp'] = [self.lp_min.value(), self.lp_max.value()]
        self.pb.taskarg['sg_win_lens'] = self.sgWin.value()
        self.pb.taskarg['highspeed'] = self.highspeed.isChecked()
        self.pb.taskarg['include_last'] = self.includeLast.isChecked()
        if self.pb.state:
            self.pb.reset()
            self.myWin.displace_result(range_fix=self.myWin.zoomfix_state)
    def spinbox_changevalue(self, value):
        sender = self.sender()
        if sender == self.sens:
            self.pb.taskarg['sens'] = value
        elif sender == self.xlimit:
            self.pb.taskarg['xlim'] = value
        elif sender == self.xsens:
            self.pb.taskarg['xsens'] = value
        elif sender == self.peakH:
            self.pb.taskarg['peakH'] = value
        elif sender == self.spinBox:
            self.myWin.siglestep = value
        elif sender == self.lp_max:
            self.pb.taskarg['lp'][1] = value
        elif sender == self.lp_min:
            self.pb.taskarg['lp'][0]=value
        elif sender == self.sgWin:
            self.pb.taskarg['sg_win_lens'] = value
    def hispeedcorrect(self):
        if self.highspeed.isChecked()==True:
            self.pb.taskarg['highspeed']=True
        else:
            self.pb.taskarg['highspeed']=False
    def cbchange(self):
        sender = self.sender()
        if sender == self.rotatestate:
            self.myWin.allowrotate = not self.myWin.allowrotate
        elif sender == self.delay:
            self.myWin.F.overlaymode = not self.myWin.F.overlaymode
        elif sender == self.includeLast:
            self.pb.taskarg['include_last'] = self.includeLast.isChecked()
class dlcrange_window(QDialog,Ui_dlc_range):
    def __init__(self,myWin):
        super(dlcrange_window, self).__init__()
        self.setupUi(self)
        self.myWin = myWin
        self.action_init()
    def action_init(self):
        self.pushButton.clicked.connect(self.table_update)
    def table_update(self):
        dic = {}
        for r in range(1, 10):
            name_item = self.tableWidget.item(r, 0)
            dlc_min_item = self.tableWidget.item(r, 1)
            dlc_max_item = self.tableWidget.item(r, 2)
            if None in [name_item, dlc_min_item, dlc_max_item]:
                continue
            if not (is_number(dlc_min_item.text()) and
                    is_number(dlc_max_item.text()) and
                    float(dlc_min_item.text()) <= float(dlc_max_item.text()) and
                    float(dlc_min_item.text()) > 0):
                QMessageBox.information(self, "Error", "Input error!")
                return None
            min_dlc = float(dlc_min_item.text())
            max_dlc = float(dlc_max_item.text())
            min_f, max_f = None, None
            force_min_item = self.tableWidget.item(r, 3)
            force_max_item = self.tableWidget.item(r, 4)
            if force_min_item is not None and force_min_item.text().strip() != '':
                if not is_number(force_min_item.text()):
                    QMessageBox.information(self, "Error", "Force min must be a number!")
                    return None
                min_f = float(force_min_item.text())
            if force_max_item is not None and force_max_item.text().strip() != '':
                if not is_number(force_max_item.text()):
                    QMessageBox.information(self, "Error", "Force max must be a number!")
                    return None
                max_f = float(force_max_item.text())
            if min_f is not None and max_f is not None and min_f >= max_f:
                QMessageBox.information(self, "Error", "min Force must be < max Force!")
                return None
            dic[name_item.text()] = (min_dlc, max_dlc, min_f, max_f)

        has_force = {k: v for k, v in dic.items() if v[2] is not None or v[3] is not None}
        no_force = {k: v for k, v in dic.items() if v[2] is None and v[3] is None}
        for k1, v1 in has_force.items():
            for k2, v2 in has_force.items():
                if k1 >= k2:
                    continue
                dlc_overlap = not (v1[1] < v2[0] or v2[1] < v1[0])
                f1_lo = v1[2] if v1[2] is not None else 0
                f1_hi = v1[3] if v1[3] is not None else 1e10
                f2_lo = v2[2] if v2[2] is not None else 0
                f2_hi = v2[3] if v2[3] is not None else 1e10
                force_overlap = not (f1_hi < f2_lo or f2_hi < f1_lo)
                if dlc_overlap and force_overlap:
                    QMessageBox.information(self, "Error",
                        '"{}" and "{}" overlap in both dLc and Force!'.format(k1, k2))
                    return None
        for k1, v1 in no_force.items():
            for k2, v2 in no_force.items():
                if k1 >= k2:
                    continue
                if not (v1[1] < v2[0] or v2[1] < v1[0]):
                    QMessageBox.information(self, "Error",
                        '"{}" and "{}" overlap in dLc with no force filter!'.format(k1, k2))
                    return None
        self.myWin.pb.taskarg['mark'] = dic
        pb = self.myWin.pb
        if pb.state and pb.tasktype == 'smfs' and len(pb.fc.data.get('peakindex', [])) > 0:
            pb.fc.recover_force(pb.ljp)
            from src.datapro import countdlc, mkbaseondlc
            countdlc(pb.fc)
            mkbaseondlc(pb.fc)
            pb.curve_change()
        self.close()
class showimage(QDialog,Ui_image):
    def __init__(self):
        super(showimage, self).__init__()
        self.setupUi(self)
    def show_img(self,img):
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
class scatterFigure(FigureCanvas):
    def __init__(self):
        #self.figure = mpl.figure.Figure()
        self.canvas = FigureCanvas(mpl.figure.Figure(dpi=100))
        #self.figure = self.canvas.figure
        self.ax = self.canvas.figure.add_subplot(4,1,(2,4))
        self.ax0 = self.canvas.figure.add_subplot(4,1,(1,1))
        
        plt.subplots_adjust(left=0.3, bottom=0.2, right=0.9, top=0.9,hspace=0,wspace=0)
        #self.figure.patch.set_facecolor('None')
        #self.figure.patch.set_alpha(0)
        self.index = 0
        self.content = {'curve': [],
                        'peak': [],
                        'bottom': [],
                        'mark': [],
                        'fitcurve': [],
                        'k':[],
                        'selrange':[]}
        self.range_fix = False
        self.s = []
        self.h = []
        super(scatterFigure, self).__init__(self.canvas.figure)
    def plotscatter(self,arr_dic,index):
        for s in self.s:
            s.remove()
        self.s = []
        x = [v[0] for i,v in arr_dic.items() if i!=index]
        y = [v[1] for i,v in arr_dic.items() if i!=index]
        if 0 not in [len(x),len(y)]:
            arr_x = np.hstack(x)
            arr_y = np.hstack(y)
            self.s.append(self.ax.scatter(arr_x,arr_y,c='k'))
        x = [v[0] for i,v in arr_dic.items() if i==index]
        y = [v[1] for i,v in arr_dic.items() if i==index]
        if 0 not in [len(x),len(y)]:
            arr_x = np.hstack(x)
            arr_y = np.hstack(y)
            self.s.append(self.ax.scatter(arr_x,arr_y,c='r'))
    def plothisto(self,arr_dic):
        for h in self.h:
            h.remove()
        x = [v[0] for i,v in arr_dic.items()]
        if 0 not in [len(x)]:
            arr_x = np.hstack(x)
            self.s.append(self.ax0.hist(arr_x,bins=10)[-1])
    def clean(self):
        for l in self.ax.lines:
            l.remove()
class statistics_win(QDialog,Ui_hist_scatter):
    def __init__(self,myWin):
        super(statistics_win, self).__init__()
        self.myWin = myWin
        self.setupUi(self)
        self.F = scatterFigure()
        self.horizontalLayout_2.addWidget(self.F.canvas)
        self.force_index = 0
        self.arr_dic = {}
        self.displace()
        self.action_init()
    def action_init(self):
        self.pushButton.clicked.connect(self.delete)
        self.pushButton_4.clicked.connect(self.plot)
        self.pushButton_5.clicked.connect(self.save)
        self.pushButton_6.clicked.connect(self.openfile)
    def displace(self):
        self.F.clean()
        self.horizontalLayout_2.removeWidget(self.F.canvas)
        self.F.canvas.draw()
        self.horizontalLayout_2.addWidget(self.F.canvas)
    def get_data(self):
        if not self.myWin.pb.state and self.myWin.pb.tasktype!='smfs':
            return None
        fc = self.myWin.pb.fc
        ljp = self.myWin.pb.ljp
        self.force_index = self.myWin.pb.forcecurve_index
        fc.recover_force(ljp)
        data = fc.get_prodata()['retract']
        data_y = data['vDeflection'].reshape(-1)*1e12
        force_arr = data_y[fc.data['peakindex'][:-1]]
        print(force_arr)
        dlc_arr = fc.data['dlc']
        print(dlc_arr)
        self.arr_dic[self.force_index] = (dlc_arr,force_arr)
        fc.clean_force()
        self.fname = 'test.scatterplot'
    def plot(self):
        if not self.myWin.pb.state or self.myWin.pb.tasktype!='smfs':
            return None
        self.get_data()
        self.F.plotscatter(self.arr_dic,self.force_index)
        self.F.plothisto(self.arr_dic)
        self.displace()
    def delete(self):
        self.force_index = self.myWin.pb.forcecurve_index
        if self.force_index in self.arr_dic.keys():
            del self.arr_dic[self.force_index]
        self.F.plotscatter(self.arr_dic,self.force_index)
        self.F.plothisto(self.arr_dic)
        self.displace()
    def save(self):
        import os
        import pickle
        if not self.myWin.pb.state or self.myWin.pb.tasktype!='smfs':
            return None
        fname = self.myWin.pb.zpo.fname
        todir = os.path.dirname(fname)
        basename = os.path.basename(fname)
        rawname = os.path.splitext(basename)[0]
        fname = os.path.join(todir,"{}.{}".format(rawname,'scatterplot'))
        with open(fname,'wb') as f:
            data = dict(arr_dic=self.arr_dic)
            pickle.dump(data,f)
    def openfile(self):
        import os
        import pickle
        if not self.myWin.pb.state or self.myWin.pb.tasktype!='smfs':
            return None
        fname = self.myWin.pb.zpo.fname
        todir = os.path.dirname(fname)
        basename = os.path.basename(fname)
        rawname = os.path.splitext(basename)[0]
        fname = os.path.join(todir,"{}.{}".format(rawname,'scatterplot'))
        if os.path.isfile(fname):
            pass
        else:
            fname, _ = QFileDialog.getOpenFileName(self, "Open Scatter Plot", '*.scatterplot')
        if os.path.isfile(fname):
            with open(fname,'rb') as f:
                data = pickle.load(f)
                self.arr_dic = data['arr_dic']
            self.F.plotscatter(self.arr_dic,self.force_index)
            self.F.plothisto(self.arr_dic)
            self.displace()
        pass
class script_win(QDialog,Ui_Script):
    def __init__(self,myWin):
        super(script_win, self).__init__()
        self.setupUi(self)
        self.myWin = myWin
        self.script_path = r'./scripts'
        self.get_script()
        self.renew_list2()
        self.listWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listWidget_2.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.add_lst = []
        self.action_init()
    def action_init(self):
        self.pushButton.clicked.connect(self.add)
        self.pushButton_2.clicked.connect(self.delete)
        self.pushButton_3.clicked.connect(self.get_selectitem)
        self.pushButton_4.clicked.connect(self.quickstart)
        self.pushButton_5.clicked.connect(self.up)
        self.pushButton_6.clicked.connect(self.down)
        self.pushButton_7.clicked.connect(self.renew)
        self.pushButton_8.clicked.connect(self.get_path)
    def start(self,item):
        pass
    def delete(self):
        self.get_selectitem()
        print(self.select_dic['add'])
        print(self.add_lst)
        for i in self.select_dic['add']:
            self.listWidget.removeItemWidget(self.listWidget.takeItem(self.listWidget.row(i)))
            self.add_lst.remove(i.text())
    def add(self):
        self.get_selectitem()
        for i in self.select_dic['script']:
            self.add_lst.append(i.text())
        self.renew_list1()
    def up(self):
        self.move(-1)
    def down(self):
        self.move(+1)
    def move(self,n):
        self.get_selectitem()
        for i,v in enumerate(self.select_dic['add']):
            index = self.listWidget.row(v)
            try:
                self.add_lst[index],self.add_lst[index+n] = self.add_lst[index+n],self.add_lst[index]
            except:
                pass
            break
        self.renew_list1()
    def get_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Load batch of force curve', '*.*')
        if path != '':
            self.script_path = path
    def get_selectitem(self):
        self.select_dic = {}
        dic = {'add':self.listWidget,'script':self.listWidget_2}
        for i in ['add','script']:
            self.select_dic[i]=[]
            items = dic[i].selectedItems()
            for item in items:
                self.select_dic[i].append(item)
    def renew(self):
        self.listWidget_2.clear()
        self.get_script()
        self.renew_list2()
    def renew_list1(self):
        self.listWidget.clear()
        for i in self.add_lst:
            self.listWidget.addItem(i)
    def renew_list2(self):
        for fname in self.f_lst:
            self.listWidget_2.addItem(fname)
        pass
    def get_script(self):
        import os
        self.f_lst = []
        for a,b,c in os.walk(self.script_path):
            for fname in c:
                if fname.endswith('.py'):
                    self.f_lst.append(fname)
    def quickstart(self):
        import os,sys
        zpo,ljp,fc = self.myWin.pb.zpo,self.myWin.pb.ljp,self.myWin.pb.fc
        index_lst = range(len(zpo))
        for i,v in enumerate(self.add_lst):
            print(v)
            name = os.path.splitext(v)[0]
            print(name)
            exec("from scripts.{} import {}".format(name,name))
            reload(sys.modules['scripts'])
            reload(sys.modules['scripts.{}'.format(name)])
            exec("from scripts.{} import {}".format(name,name))
            exec("{}_{} = {}(zpo,ljp,fc)".format(name,i,name))
        for index in index_lst:
            for i,v in enumerate(self.add_lst):
                name = os.path.splitext(v)[0]
                try:
                    exec("{}_{}.run({})".format(name,i,index))
                except Exception as err:
                    print(err)
        for i,v in enumerate(self.add_lst):
            name = os.path.splitext(v)[0]
            try:
                exec("{}_{}.end()".format(name,i))
            except Exception as err:
                print(err)
        print('finish')
        

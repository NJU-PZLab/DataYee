# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:01:41 2022

@author: zbin
"""
import sys
import copy
import os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QProgressDialog, QGridLayout, \
    QButtonGroup,QDialog,QGraphicsScene,QGraphicsPixmapItem,QTableWidgetItem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mplstyle
from plot.forcecurve_ui import Ui_MainWindow
from plot.plot_main import forcecurve_fig


class myFigure(FigureCanvas):
    def __init__(self):
        self.canvas = FigureCanvas(mpl.figure.Figure(figsize=(1,1),dpi=100))
        self.ax = self.canvas.figure.add_subplot()
        plt.subplots_adjust(hspace=0.1)
        super(myFigure, self).__init__(self.canvas.figure)
        self.ax.set_facecolor('none')
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.F = myFigure()
        self.gridlayout = QGridLayout(self.groupBox)
        self.gridlayout.addWidget(self.F.canvas)
        self.fcfig = forcecurve_fig()
        self.action_init()
    def action_init(self):
        self.fcplot.triggered.connect(self.openfc)
        self.pushButton.clicked.connect(self.plot)
        self.doubleSpinBox_4.valueChanged.connect(self.spinbox_changevalue)
    def openfc(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load force curve plot","*.plotfc")
        if fname != '':
            self.fcfig.open_(fname)
    def plot(self):
        self.fcfig.clean_axes()
        self.fcfig.create_axes(4,self.F.ax)
        self.fcfig.plot()
        self.draw()
    def draw(self):
        self.gridlayout.removeWidget(self.F.canvas)
        self.F.canvas.draw()
        self.gridlayout.addWidget(self.F.canvas)
    def spinbox_changevalue(self, value):
        sender = self.sender()
        if sender == self.doubleSpinBox_4:
            self.fcfig.plotdata['hspace'] = value
            plt.subplots_adjust(hspace=value)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
    plt.close()

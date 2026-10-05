# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scatter_histogramm.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_hist_scatter(object):
    def setupUi(self, hist_scatter):
        hist_scatter.setObjectName("hist_scatter")
        hist_scatter.resize(832, 724)
        self.horizontalLayout = QtWidgets.QHBoxLayout(hist_scatter)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(hist_scatter)
        self.groupBox.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox.setMaximumSize(QtCore.QSize(1000, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(hist_scatter)
        self.groupBox_2.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox_2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox_3.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 0, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton.setGeometry(QtCore.QRect(30, 40, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_4.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox_4.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 10, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_6.setGeometry(QtCore.QRect(30, 50, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout.addWidget(self.groupBox_4)
        self.horizontalLayout.addWidget(self.groupBox_2)

        self.retranslateUi(hist_scatter)
        QtCore.QMetaObject.connectSlotsByName(hist_scatter)

    def retranslateUi(self, hist_scatter):
        _translate = QtCore.QCoreApplication.translate
        hist_scatter.setWindowTitle(_translate("hist_scatter", "Dialog"))
        self.groupBox.setTitle(_translate("hist_scatter", "PLOT"))
        self.pushButton_4.setText(_translate("hist_scatter", "Plot"))
        self.pushButton.setText(_translate("hist_scatter", "Delete"))
        self.pushButton_5.setText(_translate("hist_scatter", "Save"))
        self.pushButton_6.setText(_translate("hist_scatter", "Open"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pairwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lb_image1 = QtWidgets.QLabel(Form)
        self.lb_image1.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_image1.setObjectName("lb_image1")
        self.horizontalLayout.addWidget(self.lb_image1)
        self.lb_image2 = QtWidgets.QLabel(Form)
        self.lb_image2.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_image2.setObjectName("lb_image2")
        self.horizontalLayout.addWidget(self.lb_image2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.lb_image1.setText(_translate("Form", "image1"))
        self.lb_image2.setText(_translate("Form", "Image2"))


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
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lb_image2 = QtWidgets.QLabel(Form)
        self.lb_image2.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_image2.setObjectName("lb_image2")
        self.verticalLayout_2.addWidget(self.lb_image2)
        self.lb_path1 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_path1.sizePolicy().hasHeightForWidth())
        self.lb_path1.setSizePolicy(sizePolicy)
        self.lb_path1.setObjectName("lb_path1")
        self.verticalLayout_2.addWidget(self.lb_path1)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lb_image1 = QtWidgets.QLabel(Form)
        self.lb_image1.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_image1.setObjectName("lb_image1")
        self.verticalLayout.addWidget(self.lb_image1)
        self.lb_path2 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_path2.sizePolicy().hasHeightForWidth())
        self.lb_path2.setSizePolicy(sizePolicy)
        self.lb_path2.setObjectName("lb_path2")
        self.verticalLayout.addWidget(self.lb_path2)
        self.verticalLayout_3.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.lb_image2.setText(_translate("Form", "Image2"))
        self.lb_path1.setText(_translate("Form", "path1"))
        self.lb_image1.setText(_translate("Form", "image1"))
        self.lb_path2.setText(_translate("Form", "path2"))


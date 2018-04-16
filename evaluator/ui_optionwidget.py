# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'optionwidget.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.cd_dataset = QtWidgets.QComboBox(self.groupBox)
        self.cd_dataset.setObjectName("cd_dataset")
        self.cd_dataset.addItem("")
        self.cd_dataset.addItem("")
        self.cd_dataset.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.cd_dataset)
        self.cb_examples = QtWidgets.QComboBox(self.groupBox)
        self.cb_examples.setObjectName("cb_examples")
        self.cb_examples.addItem("")
        self.cb_examples.addItem("")
        self.cb_examples.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cb_examples)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.sb_diameter = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.sb_diameter.setMinimum(0.1)
        self.sb_diameter.setMaximum(20.0)
        self.sb_diameter.setSingleStep(0.1)
        self.sb_diameter.setProperty("value", 5.0)
        self.sb_diameter.setObjectName("sb_diameter")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.sb_diameter)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.verticalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Display options"))
        self.label.setText(_translate("Form", "Dataset"))
        self.label_2.setText(_translate("Form", "Examples"))
        self.cd_dataset.setItemText(0, _translate("Form", "Train and Test"))
        self.cd_dataset.setItemText(1, _translate("Form", "Train"))
        self.cd_dataset.setItemText(2, _translate("Form", "Test"))
        self.cb_examples.setItemText(0, _translate("Form", "Positive and Negative"))
        self.cb_examples.setItemText(1, _translate("Form", "Positive"))
        self.cb_examples.setItemText(2, _translate("Form", "Negative"))
        self.label_3.setText(_translate("Form", "Point diameter"))


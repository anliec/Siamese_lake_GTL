from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from evaluator.ui_optionwidget import Ui_Form


class OptionWidget(QWidget):
    def __init__(self):
        super().__init__()
        # setup ui
        self.ui = Ui_Form()
        self.ui.setupUi(self)



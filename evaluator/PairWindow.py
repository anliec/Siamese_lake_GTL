from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from evaluator.ui_pairwindow import Ui_Form


class PairWindow(QWidget):
    def __init__(self):
        super().__init__()
        # setup ui
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # self.ui.lb_image1.setScaledContents(True)
        # self.ui.lb_image2.setScaledContents(True)
        self.pixmap1 = None
        self.pixmap2 = None

    @pyqtSlot(str, str, float)
    def set_image_pair(self, path_to_image1: str, path_to_image2: str, score: float=-1.0):
        try:
            self.pixmap1 = QPixmap(path_to_image1)
            self.pixmap2 = QPixmap(path_to_image2)
            self.ui.lb_image1.setPixmap(self.pixmap1.scaled(self.ui.lb_image1.size(),
                                                            Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))
            self.ui.lb_image2.setPixmap(self.pixmap2.scaled(self.ui.lb_image2.size(),
                                                            Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))
            self.ui.lb_path1.setText(path_to_image1)
            self.ui.lb_path2.setText(path_to_image2)
            if score != -1.0:
                self.ui.lb_score.setText("score: " + str(score))
            else:
                self.ui.lb_score.setText("")
            self.setEnabled(True)
            self.show()
        except Exception:
            print("Error when loading file", path_to_image1, "and", path_to_image2)
            self.setEnabled(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap1 is not None and self.pixmap2 is not None:
            self.ui.lb_image1.setPixmap(self.pixmap1.scaled(self.ui.lb_image1.size(),
                                                            Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))
            self.ui.lb_image2.setPixmap(self.pixmap2.scaled(self.ui.lb_image2.size(),
                                                            Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))


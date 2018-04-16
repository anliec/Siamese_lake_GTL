from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from keras.models import load_model
from threading import Thread

from evaluator.evaluate import DatasetTester
from evaluator.LakeGraphicsView import LakeGraphicsView
from evaluator.ui_mainwindow import Ui_MainWindow
from evaluator.PairWindow import PairWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        # setup ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.graphicView = LakeGraphicsView()
        self.setCentralWidget(self.graphicView)
        self.graphicView.setEnabled(False)
        # file chooser
        self.file_chooser = QFileDialog(self, "chose the model you want to test")
        self.file_chooser.setAcceptMode(QFileDialog.AcceptOpen)
        self.file_chooser.setFileMode(QFileDialog.ExistingFile)
        self.file_chooser.setNameFilter("Keras model file (*.h5)")
        self.file_chooser.fileSelected[str].connect(self.model_chosen)
        # worker thread
        self.worker = WorkerThread()
        self.worker.finished.connect(self.on_worker_job_finished)
        # pair window
        self.pair_window = PairWindow()
        # connection
        self.ui.actionOpen_siamese_model.triggered.connect(self.open_model_chooser)
        self.ui.actionQuit.triggered.connect(self.close)
        self.graphicView.pair_selected[str, str].connect(self.pair_window.set_image_pair)

    def open_model_chooser(self):
        self.file_chooser.setModal(True)
        self.file_chooser.show()
        self.graphicView.setEnabled(False)

    def model_chosen(self, path_to_model):
        self.ui.actionOpen_siamese_model.setEnabled(False)
        self.worker.set_model(path_to_model)
        self.worker.start()
        self.ui.statusbar.showMessage("Evaluating selected model")

    def on_worker_job_finished(self):
        self.ui.statusbar.showMessage("Evaluation finished, updating scene")
        self.graphicView.point_list = self.worker.result
        self.graphicView.update_scene()
        self.graphicView.setEnabled(True)
        self.ui.actionOpen_siamese_model.setEnabled(True)
        self.ui.statusbar.showMessage("Scene ready", 5000)


class WorkerThread(QThread):
    def __init__(self):
        super().__init__()
        self.dataset_tester = None
        self.model_path = None
        self.result = None

    def set_model(self, path: str):
        self.model_path = path

    def __del__(self):
        self.wait()

    def run(self):
        if self.model_path is None:
            return
        model = load_model(self.model_path)
        if self.dataset_tester is None:
            self.dataset_tester = DatasetTester("/tmp/data2")

        self.result = self.dataset_tester.evaluate(model,
                                                   mode='test',
                                                   batch_size=32,
                                                   add_coordinate=True)


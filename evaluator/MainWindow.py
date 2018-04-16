from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from keras.models import load_model
from threading import Thread

from evaluator.evaluate import DatasetTester
from evaluator.LakeGraphicsView import LakeGraphicsView


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
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

    def open_model_chooser(self):
        self.file_dialog.setModal(True)
        self.file_chooser.show()

    def model_chosen(self, path_to_model):
        model = load_model(path_to_model)
        self.worker.set_model(model)
        self.worker.start()

    def on_worker_job_finished(self):
        self.graphicView.point_list = self.worker.result
        self.graphicView.update_scene()


class WorkerThread(QThread):
    def __init__(self):
        super().__init__()
        self.dataset_tester = None
        self.model = None
        self.result = None

    def set_model(self, model):
        self.model = model

    def __del__(self):
        self.wait()

    def run(self):
        if self.model is None:
            return
        if self.dataset_tester is None:
            self.dataset_tester = DatasetTester()

        self.result = self.dataset_tester.evaluate(self.model,
                                                   mode='test',
                                                   batch_size=32,
                                                   add_coordinate=True)


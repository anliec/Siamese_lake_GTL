from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from keras.models import load_model
from threading import Thread
import pickle

from evaluator.evaluate import DatasetTester
from evaluator.LakeGraphicsView import LakeGraphicsView
from evaluator.ui_mainwindow import Ui_MainWindow
from evaluator.PairWindow import PairWindow
from evaluator.OptionWidget import OptionWidget


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
        self.file_chooser = QFileDialog(self, "chose model or precomputed results")
        self.file_chooser.setAcceptMode(QFileDialog.AcceptOpen)
        self.file_chooser.setFileMode(QFileDialog.ExistingFile)
        self.file_chooser.setNameFilters(["Keras model file (*.h5)", "Precomputed results (*.pickle)"])
        self.file_chooser.fileSelected[str].connect(self.model_chosen)
        # worker thread
        self.worker = WorkerThread()
        self.worker.finished.connect(self.on_worker_job_finished)
        # pair window
        self.pair_window = PairWindow()
        # dock widget
        self.option_widget = OptionWidget()
        self.dock_widget = QDockWidget(self.tr("Options"), self)
        self.dock_widget.setWidget(self.option_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        self.dock_widget.setVisible(True)
        self.dock_widget.setEnabled(False)
        # connection
        self.ui.actionOpen_siamese_model.triggered.connect(self.open_model_chooser)
        self.ui.actionQuit.triggered.connect(self.close)
        self.graphicView.pair_selected[str, str].connect(self.pair_window.set_image_pair)
        self.option_widget.ui.cb_examples.currentIndexChanged[str].connect(self.graphicView.set_display_examples)
        self.option_widget.ui.cd_dataset.currentIndexChanged[str].connect(self.graphicView.set_display_dataset)
        self.option_widget.ui.sb_diameter.valueChanged[float].connect(self.graphicView.set_point_diameter)

    def open_model_chooser(self):
        self.file_chooser.setModal(True)
        self.file_chooser.show()
        self.graphicView.setEnabled(False)
        self.dock_widget.setEnabled(False)

    def model_chosen(self, path_to_model):
        if path_to_model[-3:] == ".h5":
            self.ui.actionOpen_siamese_model.setEnabled(False)
            self.worker.set_model(path_to_model)
            self.worker.start()
            self.ui.statusbar.showMessage("Evaluating selected model")
        elif path_to_model[-7:] == ".pickle":
            try:
                self.graphicView.point_list = pickle.load(path_to_model)
                self.graphicView.update_scene()
                self.graphicView.setEnabled(True)
                self.dock_widget.setEnabled(True)
                self.ui.actionOpen_siamese_model.setEnabled(True)
                self.ui.statusbar.showMessage("Scene ready", 5000)
            except TypeError:
                self.ui.actionOpen_siamese_model.setEnabled(True)
                self.ui.statusbar.showMessage("Failed to load pickle file", 5000)

    def on_worker_job_finished(self):
        self.ui.statusbar.showMessage("Evaluation finished, updating scene")
        self.graphicView.point_list = self.worker.result
        self.graphicView.update_scene()
        self.graphicView.setEnabled(True)
        self.dock_widget.setEnabled(True)
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
                                                   mode='both',
                                                   batch_size=32,
                                                   add_coordinate=True)


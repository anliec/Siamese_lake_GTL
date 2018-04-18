from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


POINT_DIAMETER = 5.0


class LakeGraphicsView(QGraphicsView):
    pair_selected = pyqtSignal([str, str])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_list = []
        self.lake_scene = QGraphicsScene()
        self.setScene(self.lake_scene)
        self.display_train = True
        self.display_test = True
        self.display_positive = True
        self.display_negative = True
        self.diameter = POINT_DIAMETER

    def set_point_list(self, point_list):
        self.point_list = point_list

    def update_scene(self):
        scene = self.lake_scene
        scene.clear()
        scene.setBackgroundBrush(QBrush(QColor(100, 100, 100)))
        max_x, min_x, max_y, min_y = -float("Inf"), float("Inf"), -float("Inf"), float("Inf")
        for file_left, file_right, score, x, y in self.point_list:
            path = file_left.split('/')
            if len(path) < 4:
                print("Warning: path", file_left, "is shorter than expected")
                continue
            # continue if the options specify that the point must not by displayed
            if (path[-2] == "1" and not self.display_positive) or (path[-2] == "-1" and not self.display_negative):
                continue
            if (path[-4] == "train" and not self.display_train) or (path[-4] == "test" and not self.display_test):
                continue
            brush = QBrush(QColor(int(255 * (1.0 - score)), int(255 * score), 0))
            pen = QPen(brush, 1.0)
            fx, fy = float(x), -float(y)
            scene.addEllipse(fx, fy, self.diameter, self.diameter, pen, brush)
            max_x, min_x = max(max_x, fx), min(min_x, fx)
            max_y, min_y = max(max_y, fy), min(min_y, fy)
        scene.setSceneRect(min_x - POINT_DIAMETER, min_y - POINT_DIAMETER,
                           max_x - min_x + 2 * POINT_DIAMETER, max_y - min_y + 2 * POINT_DIAMETER)
        scene.update(scene.sceneRect())
        self.setScene(scene)

        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        point = self.mapToScene(event.pos())
        for file_left, file_right, score, x, y in self.point_list:
            fx, fy = float(x), -float(y)
            if fx <= point.x() <= fx + self.diameter and fy <= point.y() <= fy + self.diameter:
                self.pair_selected.emit(file_left, file_right)
                return

    def resizeEvent(self, event):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    @pyqtSlot(str)
    def set_display_dataset(self, selection_message: str):
        if selection_message == "Train":
            self.display_test = False
            self.display_train = True
        elif selection_message == "Test":
            self.display_test = True
            self.display_train = False
        elif selection_message == "Train and Test":
            self.display_test = True
            self.display_train = True

        self.update_scene()

    @pyqtSlot(str)
    def set_display_examples(self, selection_message: str):
        if selection_message == "Positive":
            self.display_negative = False
            self.display_positive = True
        elif selection_message == "Negative":
            self.display_negative = True
            self.display_positive = False
        elif selection_message == "Positive and Negative":
            self.display_negative = True
            self.display_positive = True

        self.update_scene()

    @pyqtSlot(float)
    def set_point_diameter(self, new_diameter: float):
        self.diameter = new_diameter
        self.update_scene()


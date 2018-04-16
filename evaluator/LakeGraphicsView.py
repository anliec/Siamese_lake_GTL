from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class LakeGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_list = []
        self.lake_scene = QGraphicsScene()
        self.setScene(self.lake_scene)

    def set_point_list(self, point_list):
        self.point_list = point_list

    def update_scene(self):
        scene = self.lake_scene
        scene.clear()
        scene.setBackgroundBrush(QBrush(QColor(100, 100, 100)))
        max_x, min_x, max_y, min_y = -float("Inf"), float("Inf"), -float("Inf"), float("Inf")
        for file_left, file_right, score, x, y in self.point_list:
            brush = QBrush(QColor(int(255 * (1.0 - score)), int(255 * score), 0))
            pen = QPen(brush, 1.0)
            fx, fy = -float(x), -float(y)
            scene.addEllipse(fx, fy, 10.0, 10.0, pen, brush)
            max_x, min_x = max(max_x, fx), min(min_x, fx)
            max_y, min_y = max(max_y, fy), min(min_y, fy)
        scene.setSceneRect(min_x - 1.0, min_y - 1.0, max_x - min_x + 12.0, max_y - min_y + 12.0)
        scene.update(scene.sceneRect())
        self.setScene(scene)

        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)





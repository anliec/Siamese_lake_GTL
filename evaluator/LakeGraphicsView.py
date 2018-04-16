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
        scene = self.scene()
        scene.clear()
        max_x, min_x, max_y, min_y = 0.0, 0.0, 0.0, 0.0
        for file_left, file_right, score, x, y in self.point_list:
            brush = QBrush(QColor(int(255 * (1.0 - score)), int(255 * score), 0))
            pen = QPen(brush, 0.2)
            scene.addElipse(float(x), float(y), 1.0, 1.0, pen, brush)
            max_x, min_x = max(max_x, x), min(min_x, x)
            max_y, min_y = max(max_y, y), min(min_y, y)
        scene.setSceneRect(min_x - 1.0, min_y - 1.0, max_x - min_x + 2.0, max_y - min_y + 2.0)
        scene.update(scene.sceneRect())

        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)





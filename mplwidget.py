# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes1 = self.canvas.figure.add_subplot(211)
        self.canvas.axes2 = self.canvas.figure.add_subplot(212)
        self.canvas.axes3 = self.canvas.figure.add_subplot(111)

        self.canvas.axes1.set_visible(False)
        self.canvas.axes2.set_visible(False)
        self.canvas.axes3.set_visible(False)
        self.setLayout(vertical_layout)
# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.4.4                                         #
# Author : W-yt                                             #
# File   : window                                           #
# ######################################################### #

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal
from gui import Ui_MainWindow
from vision import VersionThread

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        self.setupUi(self)

        # create the vision thread
        self.vision_thread = VersionThread()

        # connect the signal and slot
        self.actionstart.triggered.connect(self.chess_start)
        self.actionstop.triggered.connect(self.chess_stop)

        self.vision_thread.board_signal.connect(self.board_getted)
        self.vision_thread.piece_signal.connect(self.piece_update)


    def chess_start(self):
        # start the vision thread
        self.vision_thread.start()
        print("vision thread start!")


    def chess_stop(self):
        self.vision_thread.stop()
        print("vision thread stop!")


    def board_getted(self):
        print("board detect finish!")


    def piece_update(self):
        pass


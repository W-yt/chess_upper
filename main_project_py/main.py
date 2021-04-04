# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.4.3                                         #
# Author : W-yt                                             #
# File   : main                                             #
# ######################################################### #

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from window import Window

if __name__ == "__main__":
    # get the commend argument
    app = QApplication(sys.argv)
    # get the Window object and display
    display_window = Window()
    display_window.setObjectName("display_window")
    display_window.setStyleSheet("#display_window{border-image:url(image/chessboard.jpg)}")
    display_window.show()

    # enter the event loop
    sys.exit(app.exec_())


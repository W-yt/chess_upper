# ######################################################### #
# Chinese chess robot upper-computer main project by python #
# Create : 2021.4.3                                         #
# Author : W-yt                                             #
# File   : gui                                              #
# ######################################################### #

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(750, 678)

        Form.setPixmap(QtGui.QPixmap("image/chessboard.jpg"))


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

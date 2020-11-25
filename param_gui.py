import sys,os,json

from PyQt5.QtWidgets import QComboBox, QLabel, QGroupBox, QMainWindow, QWidget, QComboBox, QLineEdit, QVBoxLayout, QHBoxLayout, QApplication, QPushButton, qApp
from PyQt5 import QtCore

class Param_GUI(QMainWindow):
    """
    GUI for choosing training parameters
    """
    def __init__(self):
        super().__init__()
        self.main_w()

    def  main_w(self):
        self.main_widget = QWidget(self)

        self.l_fileDir = QLabel("File directory: ")
        self.l_t1MapDir = QLabel("T1 map directory: ")
        self.l_modelName = QLabel("Model name: ")
        self.l_load = QLabel("Load preset data splits: ")
        self.l_lr = QLabel("Learning rate (e-6): ")
        self.l_b1 = QLabel("Beta 1 for Adam (e-2): ")
        self.l_batchSize = QLabel("Batch size: ")
        self.l_numEpochs = QLabel("Number of epochs: ")
        self.l_stepSize = QLabel("Step size for lr scheduler: ")

        self.le_fileDir = QLineEdit("C:/fully_split_data/")
        self.le_t1MapDir = QLineEdit("C:/T1_Maps/")
        self.le_modelName = QLineEdit()
        self.cb_load = QComboBox()
        self.cb_load.addItems(["True","False"])
        self.s_lr = QLineEdit("1e-3")
        self.s_b1 = QLineEdit("0.5")
        self.s_batchSize = QLineEdit("4")
        self.s_numEpochs =  QLineEdit("50")
        self.s_stepSize =  QLineEdit("5")

        self.pb_confirm = QPushButton("Confirm and Exit")
        self.pb_confirm.clicked.connect(self.confirm_n_close)

        self.v_box_1 = QVBoxLayout()
        self.v_box_1.addWidget(self.l_fileDir)
        self.v_box_1.addWidget(self.l_t1MapDir)
        self.v_box_1.addWidget(self.l_modelName)
        self.v_box_1.addWidget(self.l_load)
        self.v_box_1.addWidget(self.l_lr)
        self.v_box_1.addWidget(self.l_b1)
        self.v_box_1.addWidget(self.l_batchSize)
        self.v_box_1.addWidget(self.l_numEpochs)
        self.v_box_1.addWidget(self.l_stepSize)

        self.v_box_2 = QVBoxLayout()
        self.v_box_2.addWidget(self.le_fileDir)
        self.v_box_2.addWidget(self.le_t1MapDir)
        self.v_box_2.addWidget(self.le_modelName)
        self.v_box_2.addWidget(self.cb_load)
        self.v_box_2.addWidget(self.s_lr)
        self.v_box_2.addWidget(self.s_b1)
        self.v_box_2.addWidget(self.s_batchSize)
        self.v_box_2.addWidget(self.s_numEpochs)
        self.v_box_2.addWidget(self.s_stepSize)

        self.h_box = QHBoxLayout()
        self.h_box.addLayout(self.v_box_1)
        self.h_box.addLayout(self.v_box_2)
        self.h_box.addWidget(self.pb_confirm)

        self.main_widget.setLayout(self.h_box)
        self.setCentralWidget(self.main_widget)
        self.show()

    def confirm_n_close(self,*args,**kwargs):
        hParamDict = {}
        hParamDict["fileDir"] = self.le_fileDir.text()
        hParamDict["t1MapDir"] = self.le_t1MapDir.text()
        hParamDict["modelName"] = self.le_modelName.text()
        if self.cb_load.currentText() == "False":
            hParamDict["load"] = False
        else:
            hParamDict["load"] = True
        hParamDict["lr"] = float(self.s_lr.text())
        hParamDict["b1"] = float(self.s_b1.text())
        hParamDict["batchSize"] = int(self.s_batchSize.text())
        hParamDict["numEpochs"] = int(self.s_numEpochs.text())
        hParamDict["stepSize"] = int(self.s_stepSize.text())

        os.makedirs("./TrainingLogs/{}/".format(hParamDict["modelName"]))

        with open("./TrainingLogs/{}/hparams.json".format(hParamDict["modelName"]),"w") as f:
            json.dump(hParamDict,f)
        self.close()


def main():
    app = QApplication(sys.argv)
    main_win = Param_GUI()
    main_win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
import sys
from PyQt5.QtWidgets import QApplication, QWidget

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.move(50, 50)
        self.resize(1500, 800)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = HandSeals()
    sys.exit(app.exec_())
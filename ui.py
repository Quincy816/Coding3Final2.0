
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from predict import main1
from PIL import Image

#learned from https://github.com/qq1308636759/VGG16--/blob/main/UI.py.
class Ui_example(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QGridLayout(self)
        self.label_image = QLabel(self)
        self.label_predict_result = QLabel('Result',self)
        self.label_predict_result_display = QLabel(self)
        self.label_flower_meaning = QLabel('Language of the flower', self)
        self.label_flower_meaning_display = QLabel(self)

        self.button_search_image = QPushButton('Upload',self)
        self.button_run = QPushButton('Recognize',self)


        self.setLayout(self.layout)
        self.initUi()

    def initUi(self):

        self.layout.addWidget(self.label_image,1,1,8,4)
        self.layout.addWidget(self.button_search_image,1,8,1,2)
        self.layout.addWidget(self.button_run,3,8,1,2)
        self.layout.addWidget(self.label_predict_result,6,8,1,1)
        self.layout.addWidget(self.label_predict_result_display,6,9,1,1)
        self.layout.addWidget(self.label_flower_meaning, 8, 8, 1, 1)
        self.layout.addWidget(self.label_flower_meaning_display, 8, 9, 1, 1)

        self.button_search_image.clicked.connect(self.openimage)

        self.button_run.clicked.connect(self.run)


        self.setGeometry(300,300,1000,600)
        self.setWindowTitle('Flower recognition system based on deep learning convolutional neural networks')
        self.show()

    def openimage(self):

        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "select", "", "*.jpg;;*.jpeg;;*.png;;All Files(*)")

        jpg = QPixmap(imgName).scaled(self.label_image.width(), self.label_image.height())


        self.label_image.setPixmap(jpg)
        fname = imgName

    def run(self):
        global fname
        file_name = str(fname)
        img = Image.open(file_name)

        predicted_flower, _ = main1(img)  # 忽略准确率
        self.label_predict_result_display.setText(predicted_flower)
        flower_meanings = {
            "daisy": "Pure beauty, innocence, childishness, happiness, peace, 'love deep in the heart'.",
            "dandelion": "I pray for your happiness from a distance.",
            "roses": "Pure love, beautiful love, beauty always exists.",
            "sunflowers": "Silent, unspoken love.",
            "tulips": "Fraternity, consideration, elegance, wealth, ability, intelligence."
        }

        meaning = flower_meanings.get(predicted_flower, "no")
        self.label_flower_meaning_display.setText(meaning)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ui_example()
    sys.exit(app.exec_())
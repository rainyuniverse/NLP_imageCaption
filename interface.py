import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import caption
import argparse
import torch
import json

class filedialogdemo(QWidget):

    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        layout = QVBoxLayout()

        self.btn = QPushButton()
        self.btn.clicked.connect(self.loadFile)
        self.btn.setText("从文件中获取照片")
        layout.addWidget(self.btn)

        self.label = QLabel()
        layout.addWidget(self.label)

        # self.btn_2 = QPushButton()
        # self.btn_2.clicked.connect(self.load_text)
        # self.btn_2.setText("显示文本")
        # layout.addWidget(self.btn_2)

        self.content = QTextEdit()
        layout.addWidget(self.content)
        self.setWindowTitle("测试")

        self.setLayout(layout)

    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'c:\\', 'Image files(*.jpg *.gif *.png)')
        print(fname)
        self.label.setPixmap(QPixmap(fname))

        defult_path = './BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth'
        word_map_path = './WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
        img_path = fname
        parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
        parser.add_argument('--img', '-i', default=img_path, help='path to image')
        parser.add_argument('--model', '-m', default=defult_path, help='path to model')
        parser.add_argument('--word_map', '-wm', default=word_map_path, help='path to word map JSON')
        parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
        parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

        args = parser.parse_args()

        # Load model
        checkpoint = torch.load(args.model, map_location=str(caption.device))
        decoder = checkpoint['decoder']
        decoder = decoder.to(caption.device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(caption.device)
        encoder.eval()

        # Load word map (word2ix)
        with open(args.word_map, 'r') as j:
            word_map = json.load(j)

        # Encode, decode with attention and beam search
        seq, alphas = caption.caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)
        print(seq)
        result_str = ''
        for i in range(len(seq)):
            # if i == 0 or i == len(seq):
            #     continue
            result_str = result_str + ' ' + (list(word_map.keys())[seq[i] - 1])
        print(result_str)
        self.content.setText(result_str)

    def load_text(self):
        print("load--text")
        self.content.setText()
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter(QDir.Files)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            with f:
                data = f.read()
                self.content.setText(data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    fileload =  filedialogdemo()
    fileload.show()
    sys.exit(app.exec_())
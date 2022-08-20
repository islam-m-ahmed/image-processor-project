from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from exam import *
from constants import *



NON_LINEAR = {
    'max': max_filter,
    'min': min_filter,
    'mean': mean_filter
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)
        self.references()

        self.init_widgets()
        self.connect_signals()

        self.show()
    
    def references(self):
        self.qCB_image: QtWidgets.QComboBox
        self.qL_img_original: QtWidgets.QLabel
        self.qL_img_output: QtWidgets.QLabel
        self.qSB_seg_manual: QtWidgets.QSpinBox
        self.qLE_seg_auto: QtWidgets.QLineEdit
        self.qPB_apply_seg_manual: QtWidgets.QPushButton
        self.qPB_apply_seg_auto: QtWidgets.QPushButton
        self.qPB_hist_equalize: QtWidgets.QPushButton
        self.qPB_contrast_stretch: QtWidgets.QPushButton
        self.qCB_kernel_size: QtWidgets.QComboBox
        self.qCB_smooth_linear: QtWidgets.QComboBox
        self.qPB_apply_smooth_linear: QtWidgets.QPushButton
        self.qCB_smooth_non_linear: QtWidgets.QComboBox
        self.qPB_apply_smooth_non_linear: QtWidgets.QPushButton
        self.qCB_edge_filter: QtWidgets.QComboBox
        self.qCB_edge_type: QtWidgets.QComboBox
        self.qPB_apply_edge_detection: QtWidgets.QPushButton
        self.qSB_baudrate: QtWidgets.QSpinBox
        self.qCB_trans_channels: QtWidgets.QComboBox
        self.qLE_trans_time: QtWidgets.QLineEdit
        self.qPB_hist_original: QtWidgets.QPushButton
        self.qPB_hist_output: QtWidgets.QPushButton
        self.qPB_hist_both: QtWidgets.QPushButton

        self.qCB_laplacian_enhanced: QtWidgets.QCheckBox
        self.qPB_apply_laplacian: QtWidgets.QPushButton
        self.qPB_apply_high_boost: QtWidgets.QPushButton
        self.qDSB_high_boost: QtWidgets.QDoubleSpinBox

        self.qVBL_hist: QtWidgets.QVBoxLayout

    def init_widgets(self):
        for key, val in EDGE_DETECTION_FILTERS:
            self.qCB_edge_filter.addItem(key, val)
        
        for key, val in EDGE_DETECTION_TYPES:
            self.qCB_edge_type.addItem(key, val)

        for key, val in KERNEL_SIZE:
            self.qCB_kernel_size.addItem(key, val)

        for key, val in SMOOTH_LINEAR_FILTERS:
            self.qCB_smooth_linear.addItem(key, val)

        for key, val in SMOOTH_NON_LINEAR_FILTERS:
            self.qCB_smooth_non_linear.addItem(key, val)

        for key, val in TRANS_TIME_CHANNELS:
            self.qCB_trans_channels.addItem(key, val)

        self.qCB_image.addItems(IMAGES['files'].keys())
        self.sig_image_index_changed()

        self.qL_img_original.paintEvent = lambda x: self.qL_img_paintEvent(0, self.qL_img_original)
        self.qL_img_output.paintEvent = lambda x: self.qL_img_paintEvent(0, self.qL_img_output)

        self.canvas = FigureCanvas(Figure())
        self.qVBL_hist.addWidget(self.canvas)

    def connect_signals(self):
        self.qCB_image.currentIndexChanged.connect(self.sig_image_index_changed)
        self.qCB_edge_filter.currentIndexChanged.connect(self.sig_edge_filter)

        self.qCB_trans_channels.currentIndexChanged.connect(self.update_transmission_time)
        self.qSB_baudrate.valueChanged.connect(self.update_transmission_time)

        self.qPB_apply_smooth_linear.clicked.connect(self.apply_smooth_linear)
        self.qPB_apply_smooth_non_linear.clicked.connect(self.apply_non_linear)
        self.qPB_apply_seg_manual.clicked.connect(self.apply_seg_manual)
        self.qPB_apply_seg_auto.clicked.connect(self.apply_seg_auto)
        self.qPB_hist_equalize.clicked.connect(self.apply_hist_eq)
        self.qPB_contrast_stretch.clicked.connect(self.apply_contrast_stretch)
        self.qPB_apply_edge_detection.clicked.connect(self.apply_edge_detection)

        self.qPB_apply_laplacian.clicked.connect(lambda t: self.apply_sharpen('laplacian'))
        self.qPB_apply_high_boost.clicked.connect(lambda t: self.apply_sharpen('high-boost'))
        self.qPB_hist_original.clicked.connect(lambda name: self.show_histogram('original'))
        self.qPB_hist_output.clicked.connect(lambda name: self.show_histogram('output'))
        self.qPB_hist_both.clicked.connect(self.show_both_histogram)

    def get_selected_image_path(self):
        image_name = self.qCB_image.currentText()
        return get_file_path(image_name)
    
    def get_selected_image_as_grayscale(self):
        img_path = self.get_selected_image_path()
        return load_grayscale(img_path)

    def convert_to_grayscale(self, pixmap):
        qimg = QtGui.QPixmap.toImage(pixmap)
        grayscale = qimg.convertToFormat(QtGui.QImage.Format_Grayscale8)
        return QtGui.QPixmap.fromImage(grayscale)

    def qL_img_paintEvent(self, event, pix: QtWidgets.QLabel):
        size = pix.size()
        painter = QtGui.QPainter(pix)
        point = QtCore.QPoint(0,0)
        scaledPix = pix.pixmap().scaled(size, QtCore.Qt.KeepAspectRatio,
            transformMode = QtCore.Qt.SmoothTransformation)
        px = (size.width() - scaledPix.width()) // 2
        py = (size.height() - scaledPix.height()) // 2
        point.setX(px)
        point.setY(py)
        pix.setFixedHeight(scaledPix.height())
        painter.drawPixmap(point, scaledPix)
        painter.drawRect(px, py, scaledPix.width(), scaledPix.height())

    def update_output_image(self, bin_img):
        print(type(bin_img))
        image = Image.fromarray(bin_img)
        rgbimg = Image.new("RGBA", image.size)
        rgbimg.paste(image)

        rgbimg.save('data/images/output.png')
        self.qL_img_output.setPixmap(QtGui.QPixmap('data/images/output.png'))

    def sig_image_index_changed(self):
        img_path = self.get_selected_image_path()
        pixmap = self.convert_to_grayscale(QtGui.QPixmap(img_path))
        self.qL_img_original.setPixmap(pixmap)
        self.qL_img_output.setPixmap(QtGui.QPixmap(IMAGES['dir'] + 'empty.png'))
        self.update_transmission_time()
    
    def sig_edge_filter(self):
        flag = self.qCB_edge_filter.currentData() != 'canny'

        self.qCB_edge_type.setEnabled(flag)
        self.qL_edge_type_title.setEnabled(flag)

    def apply_smooth_linear(self):
        size = self.qCB_kernel_size.currentData()
        filter = self.qCB_smooth_linear.currentData()
            
        cv2_img = self.get_selected_image_as_grayscale()
        output = apply_linear_filter(cv2_img, filter, size)

        self.update_output_image(output)

    def apply_non_linear(self):
        size = self.qCB_kernel_size.currentData()
        filter = self.qCB_smooth_non_linear.currentData()

        cv2_img = self.get_selected_image_as_grayscale()
        output = NON_LINEAR[filter](cv2_img, size)

        self.update_output_image(output)

    def apply_seg_manual(self):
        cv2_img = self.get_selected_image_as_grayscale()

        thresh = self.qSB_seg_manual.value()
        output = apply_manual_thres(cv2_img, thresh)

        self.update_output_image(output)

    def apply_seg_auto(self):
        cv2_img = self.get_selected_image_as_grayscale()

        output, thresh = otsu(cv2_img)
        self.qLE_seg_auto.setText(str(thresh))

        self.update_output_image(output)

    def apply_hist_eq(self):
        cv2_img = self.get_selected_image_as_grayscale()
        output = hist_eq(cv2_img)

        self.update_output_image(output)

    def apply_contrast_stretch(self):
        cv2_img = self.get_selected_image_as_grayscale()
        output = contrast_stretch(cv2_img)

        self.update_output_image(output)      

    def apply_edge_detection(self):
        edge_filter = self.qCB_edge_filter.currentData()
        edge_type = self.qCB_edge_type.currentData()
        cv2_img = self.get_selected_image_as_grayscale()

        if edge_filter != 'canny':
            if edge_type != 'both':
                output = apply_edge_detection(cv2_img, edge_filter, edge_type)
            else:
                output_1 = apply_edge_detection(cv2_img, edge_filter, 'ver')
                output_2 = apply_edge_detection(cv2_img, edge_filter, 'hor')
                output = combine_both_edge_images(output_1, output_2)
        else: # CANNY
            output = canny(cv2_img)

        self.update_output_image(output)
    
    def apply_sharpen(self, type_):
        cv2_img = self.get_selected_image_as_grayscale()

        a_val = (1 if self.qCB_laplacian_enhanced.isChecked() else 0) \
            if type_ == 'laplacian' else self.qDSB_high_boost.value()

        output = apply_sharpening_filter(cv2_img, a_val)
        self.update_output_image(output)
    
    def update_transmission_time(self):
        cv2_img = self.get_selected_image_as_grayscale()
        baudrate = self.qSB_baudrate.value()
        channels = self.qCB_trans_channels.currentData()

        time_in_sec = time_transmission(cv2_img, baudrate, channels)
        self.qLE_trans_time.setText(str(time_in_sec))

    def show_histogram(self, name):
        image = self.get_selected_image_as_grayscale() if name == 'original' \
            else load_grayscale('data/images/output.png')

        show_hist(self.canvas.figure, image)

    def show_both_histogram(self):
        original = self.get_selected_image_as_grayscale()
        output_path = 'data/images/output.png'
        output = load_grayscale(output_path)
        both_hist(self.canvas.figure, original, output)


def run_gui():
    import sys

    app = QtWidgets.QApplication(sys.argv)

    UI_EXIST = ...
    
    try:
        with open(UI_FILE) as file:
            UI_EXIST = True
        win = MainWindow()
    except FileNotFoundError:
        UI_EXIST = False
        print('Error: Important files are missing!')

    sys.exit(app.exec())


if __name__ == '__main__':
    run_gui()
import sys
import ntpath
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from functools import partial
from img_modifier import img_helper
from img_modifier import color_filter
from img_modifier import closed_form_matting
from img_modifier import solve_foreground_background
from PIL import ImageQt
from logging.config import fileConfig
import logging
import numpy as np
from PIL import Image
from scipy import misc
import tensorflow as tf
import get_dataset_colormap
from inpainter import inpainter
import cv2


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)


logger = logging.getLogger()

# original img, can't be modified
_img_original = None
_img_preview = None
_bimap = None #2D, 0 be background 1 be foregrounds
win = None

# constants
THUMB_BORDER_COLOR_ACTIVE = "#3893F4"
THUMB_BORDER_COLOR = "#ccc"
BTN_MIN_WIDTH = 120
ROTATION_BTN_SIZE = (70, 30)
THUMB_SIZE = 120

SLIDER_MIN_VAL = -100
SLIDER_MAX_VAL = 100
SLIDER_DEF_VAL = 0

PARA1_MIN_VAL = 5
PARA1_MAX_VAL = 50
PARA1_DEF_VAL = 9

PARA2_MIN_VAL = 1
PARA2_MAX_VAL = 10
PARA2_DEF_VAL = 1

IMG_DIS_W = 572
IMG_DIS_H = 335


class DeepInpaintingNet(object):
    def __init__(self, model_path = './models/ref_graph.pb'):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        with open(model_path, 'rb') as fd:
            graph_def = tf.GraphDef.FromString(fd.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

        self.x = "input:0"
        self.mask = "mask:0"	
        self.y = "gan/output:0"

    def run(self, x, mask):
        print(x.shape)
        print(mask.shape)
        assert(len(x.shape) == 3)
        assert(len(mask.shape) == 3 )
        assert(x.shape == mask.shape)
        x_in = np.expand_dims(cv2.resize(x.astype(np.float32),(256,256), interpolation=cv2.INTER_CUBIC),axis = 0 )/255
        mask_in =  np.expand_dims(cv2.resize(mask.astype(np.float32),(256,256)),axis = 0 )
        #global _img_original
        #_img_original = Image.fromarray(np.uint8(np.squeeze(mask_in)*255)) 
        result = self.sess.run(self.y, feed_dict={self.x:x_in, self.mask:mask_in})
        return result
	
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 256
    def __init__(self, model_path = './models/seg_graph.pb'):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        with open(model_path, 'rb') as fd:
            graph_def = tf.GraphDef.FromString(fd.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)
     
    def run(self, image):
        """Runs inference on a single image.
	    Args:
	        image: A PIL.Image object, raw input image.
    	Returns:
    	    resized_image: RGB image resized from original input image.
    	    seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        #target_size = (int(resize_ratio * width), int(resize_ratio * height))
        target_size = (256,256)
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={\
              self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def get_bimap(self, seg_map, image = None):
            seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
            bimap = np.equal(seg_map,0) + 0
            unique_labels = np.unique(seg_map)
            if image: 
                foregrounds = np.multiply(image, np.expand_dims(1-bimap,axis = 2))
                return [foregrounds,bimap]
            return bimap


class Operations:
    def __init__(self):
        self.color_filter = None

        self.flip_left = False
        self.flip_top = False
        self.rotation_angle = 0

        self.size = None

        self.brightness = 0
        self.sharpness = 0
        self.contrast = 0

    def reset(self):
        self.color_filter = None

        self.brightness = 0
        self.sharpness = 0
        self.contrast = 0

        self.size = None

        self.flip_left = False
        self.flip_top = False
        self.rotation_angle = 0

    def has_changes(self):
        return self.color_filter or self.flip_left\
                or self.flip_top or self.rotation_angle\
                or self.contrast or self.brightness\
                or self.sharpness or self.size

    def __str__(self):
        return f"size: {self.size}, filter: {self.color_filter}, " \
               f"b: {self.brightness} c: {self.contrast} s: {self.sharpness}, " \
               f"flip-left: {self.flip_left} flip-top: {self.flip_top} rotation: {self.rotation_angle}"


_seg_model = DeepLabModel()
_inpainting_net = DeepInpaintingNet()
operations = Operations()


def _get_ratio_height(width, height, r_width):
    return int(r_width/width*height)

def _get_ratio_width(width, height, r_height):
    return int(r_height/height*width)
def _get_converted_point(user_p1, user_p2, p1, p2, x):
    """
    convert user ui slider selected value (x) to PIL value
    user ui slider scale is -100 to 100, PIL scale is -1 to 2
    example:
     - user selected 50
     - PIL value is 1.25
    """
    r = (x - user_p1) / (user_p2 - user_p1)
    return p1 + r * (p2 - p1)


def _get_img_with_all_operations():
    logger.debug(operations)

    b = operations.brightness
    c = operations.contrast
    s = operations.sharpness

    img = _img_preview

    if b != 0:
        img = img_helper.brightness(img, b)

    if c != 0:
        img = img_helper.contrast(img, c)

    if s != 0:
        img = img_helper.sharpness(img, s)

    if operations.rotation_angle:
        img = img_helper.rotate(img, operations.rotation_angle)

    if operations.flip_left:
        img = img_helper.flip_left(img)

    if operations.flip_top:
        img = img_helper.flip_top(img)

    if operations.size:
        img = img_helper.resize(img, *operations.size)

    return img


class ActionTabs(QTabWidget):
    """Action tabs widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.filters_tab = FiltersTab(self)
        self.adjustment_tab = AdjustingTab(self)
        self.modification_tab = ModificationTab(self)
        self.seg_tab = SegTab(self)
        self.inpainting_tab = InpaintingTab(self)

        self.addTab(self.filters_tab, "Filters")
        self.addTab(self.adjustment_tab, "Adjusting")
        self.addTab(self.modification_tab, "Modification")
        self.addTab(self.seg_tab, "Character Extraction")
        self.addTab(self.inpainting_tab, "Inpainting")
        self.setMaximumHeight(190)



class SegTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
		
        self.cb_text = QLabel("Select Matting Pen Size:")
        self.cb = QComboBox()
        self.cb.addItems(["Small", "Medium", "Large"])
        self.cb.currentIndexChanged.connect(self.selectionchange)
        
        self.matting_color_text = QLabel("Select Matting Pen Color:")
        self.matting_cbw = QCheckBox('White (Foreground)', self)
        self.matting_cbw.setChecked(False)
        self.matting_cbb = QCheckBox('Black (Background)', self)
        self.matting_cbb.setChecked(False)
        self.matting_cbw.clicked.connect(self.white_clicked)
        self.matting_cbb.clicked.connect(self.black_clicked)
        self.matting_draw_btn = QPushButton("Disable Drawing")
        self.matting_draw_btn.setFixedWidth(200)
        self.matting_draw_btn.clicked.connect(self.disable_pen)
        
        self.matting_btn = QPushButton("Apply Matting")
        self.matting_btn.setFixedWidth(200)
        self.matting_btn.clicked.connect(self.matting_apply)
        self.matting_btn.setEnabled(False)

        matting_layout = QVBoxLayout()
        matting_layout.addWidget(self.cb_text)
        matting_layout.addWidget(self.cb)
        matting_layout.addWidget(self.matting_color_text)
        matting_layout.addWidget(self.matting_cbw)
        matting_layout.addWidget(self.matting_cbb)
        matting_layout.addWidget(self.matting_draw_btn)
        matting_layout.addWidget(self.matting_btn)
        matting_layout.setAlignment(Qt.AlignRight)

        self.seg_btn = QPushButton("One-tap Auto Segmentation")
        self.seg_btn.setFixedWidth(200)
        self.seg_btn.clicked.connect(self.seg_apply)

        seg_layout = QVBoxLayout()
        seg_layout.addWidget(self.seg_btn)
        seg_layout.setAlignment(Qt.AlignRight)

        groupBox_seg = PyQt5.QtWidgets.QGroupBox('Or try automatic segmentation:')
        groupBox_seg.setLayout(seg_layout)
        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(matting_layout)
        main_layout.addWidget(groupBox_seg)

        self.setLayout(main_layout)
    
    def white_clicked(self):
        self.parent.parent.drawable = True
        self.matting_btn.setEnabled(True)
        self.matting_cbw.setChecked(True)
        self.matting_cbb.setChecked(False)
        self.parent.parent.pen_color = 255

    def black_clicked(self):
        self.parent.parent.drawable = True
        self.matting_btn.setEnabled(True)
        self.matting_cbb.setChecked(True)
        self.matting_cbw.setChecked(False)
        self.parent.parent.pen_color = 0

    def disable_pen(self):
        self.parent.parent.drawable = False
        self.matting_cbw.setChecked(False)
        self.matting_cbb.setChecked(False)
          
    def selectionchange(self,i):
       if self.cb.currentText() == "Small":
           print("small!!")
           self.parent.parent.pen_size = 3
       elif self.cb.currentText() == "Medium": self.parent.parent.pen_size = 5
       elif self.cb.currentText() == "Large": self.parent.parent.pen_size = 10
       else: print("Unknown error appear when selecting pen size")

    
    def seg_apply(self, event):  
        self.seg_btn.setEnabled(False)
        self.parent.parent.parent.setEnabled(False)
        global _img_preview
        img = _img_preview
        img, seg_map = _seg_model.run(img)
        bimap = _seg_model.get_bimap(seg_map)
        global _bimap
        _bimap = bimap
        np_img = np.array(img)
        mask = np.stack((bimap,)*3, axis=-1)
        seg_output = np_img * mask
        _img_preview = Image.fromarray(np.uint8(seg_output))#.resize((_img_original.width,_img_original.height))
        self.parent.parent.place_preview_img()
        self.parent.inpainting_tab.setEnabled(True)
        self.parent.parent.parent.setEnabled(True)
      


    def matting_apply(self, event):  
        self.matting_btn.setEnabled(False)
        global _img_original, _img_preview
        original_image = np.array(_img_original)/255.0
        preview_image = np.array(_img_preview)/255.0
        global _bimap
        alpha = closed_form_matting.closed_form_matting_with_scribbles(original_image,preview_image)
        foreground, background, _ = solve_foreground_background.solve_foreground_background(np.array(_img_original),alpha)
        _bimap = (alpha <= 0.4).astype(int)
        print(_bimap.shape)
        _img_preview = Image.fromarray(np.uint8(background))
        self.parent.parent.place_preview_img()
        self.parent.inpainting_tab.setEnabled(True)



        
class InpaintingTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setEnabled(False)
        self.parent = parent

        self.para1_slider = QSlider(Qt.Horizontal, self)
        self.para1_slider.setMinimum(PARA1_MIN_VAL)
        self.para1_slider.setMaximum(PARA1_MAX_VAL)
        self.para1_slider.sliderReleased.connect(self.on_para1_change)
        self.para1_lbl = QLabel(str(self.para1_slider.value()), self)
        self.para1_lbl.setFixedWidth(20)
        self.para1_text = QLabel("Adjust Inpainting Patch Size", self)
        slider1_layout = QHBoxLayout()
        slider1_layout.addWidget(self.para1_lbl)
        slider1_layout.addWidget(self.para1_slider)
        slider1_layout.setAlignment(Qt.AlignCenter)
        para1_layout = QVBoxLayout()
        para1_layout.addWidget(self.para1_text)
        para1_layout.addLayout(slider1_layout)

        self.para2_slider = QSlider(Qt.Horizontal, self)
        self.para2_slider.setMinimum(PARA2_MIN_VAL)
        self.para2_slider.setMaximum(PARA2_MAX_VAL)
        self.para2_slider.sliderReleased.connect(self.on_para2_change)
        self.para2_lbl = QLabel(str(self.para2_slider.value()), self)
        self.para2_lbl.setFixedWidth(20)
        self.para2_text = QLabel("Adjust Inpainting Smooth Rate ", self)
        slider2_layout = QHBoxLayout()
        slider2_layout.addWidget(self.para2_lbl)
        slider2_layout.addWidget(self.para2_slider)
        slider2_layout.setAlignment(Qt.AlignCenter)
        para2_layout = QVBoxLayout()
        para2_layout.addWidget(self.para2_text)
        para2_layout.addLayout(slider2_layout)
        
        self.exemplar_btn = QPushButton("Exemplar Inpainting")
        self.exemplar_btn.setFixedWidth(180)
        self.exemplar_btn.clicked.connect(self.exemplar_apply)

        exemplar_layout = QVBoxLayout()
        exemplar_layout.addLayout(para1_layout)
        exemplar_layout.addLayout(para2_layout)
        exemplar_layout.addWidget(self.exemplar_btn)
        exemplar_layout.setAlignment(Qt.AlignCenter)


        self.ref_btn = QPushButton("Quick Inpainting")
        self.ref_btn.setFixedWidth(200)
        self.ref_btn.clicked.connect(self.coarse_to_refine_apply)

        ref_layout = QVBoxLayout()
        ref_layout.addWidget(self.ref_btn)
        ref_layout.setAlignment(Qt.AlignRight)

        groupBox_ref = PyQt5.QtWidgets.QGroupBox('Or try quick inpainting:')
        groupBox_ref.setLayout(ref_layout)
        
        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        main_layout.addLayout(exemplar_layout)
        main_layout.addWidget(groupBox_ref)

        self.reset_sliders()
        self.setLayout(main_layout)

    def reset_sliders(self):
        self.para1_slider.setValue(PARA1_DEF_VAL)
        self.para2_slider.setValue(PARA2_DEF_VAL)

    def exemplar_apply(self,event):
        self.exemplar_btn.setEnabled(False)
        self.ref_btn.setEnabled(False)
        global _img_preview
        img = _img_preview
        image = np.array(img)
        bimap = _bimap 
        output_image = inpainter.Inpainter(image, bimap, self.para1_slider.value(), self.para2_slider.value()).inpaint()
        _img_preview = Image.fromarray(np.uint8(output_image))       
        self.parent.parent.place_preview_img()

    def on_para1_change(self):
        print(self.para1_slider.value())
        self.para1_lbl.setText(str(self.para1_slider.value()))

    def on_para2_change(self):
        print(self.para2_slider.value())
        self.para2_lbl.setText(str(self.para2_slider.value()))

    def coarse_to_refine_apply(self, event):
        global _img_preview
        self.exemplar_btn.setEnabled(False)
        self.ref_btn.setEnabled(False)
        mask = np.stack((_bimap,)*3, axis=-1)

        original_shape = mask.shape 
        img = np.array(_img_preview)
        print(_bimap.shape)
        print(mask.shape)
        print(img.shape)
        inpainted_img = _inpainting_net.run(img, mask)
        print(inpainted_img.shape)
        inpainted_img = np.squeeze(inpainted_img)
        _img_preview = Image.fromarray(np.uint8(inpainted_img*255)) 
        #inpainting_img = inpainting_img.resize(original_shape)
        #SR_img = super_resolustion(inpainting_img)
        #result = merge(img_np4d, SR_img, mask_np4d)
        #_img_preview = Image.fromarray(np.uint8(result*255)) 
        self.parent.parent.place_preview_img()


class ModificationTab(QWidget):
    """Modification tab widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.width_lbl = QLabel('width:', self)
        self.width_lbl.setFixedWidth(90)

        self.height_lbl = QLabel('height:', self)
        self.height_lbl.setFixedWidth(90)

        self.width_box = QLineEdit(self)
        self.width_box.textEdited.connect(self.on_width_change)
        self.width_box.setMaximumWidth(90)

        self.height_box = QLineEdit(self)
        self.height_box.textEdited.connect(self.on_height_change)
        self.height_box.setMaximumWidth(90)

        self.unit_lbl = QLabel("px")
        self.unit_lbl.setMaximumWidth(50)

        self.ratio_check = QCheckBox('aspect ratio', self)
        self.ratio_check.stateChanged.connect(self.on_ratio_change)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedWidth(90)
        self.apply_btn.clicked.connect(self.on_apply)


        
        width_layout = QHBoxLayout()
        width_layout.addWidget(self.width_box)
        width_layout.addWidget(self.height_box)
        width_layout.addWidget(self.unit_lbl)

        apply_layout = QHBoxLayout()
        apply_layout.addWidget(self.apply_btn)
        apply_layout.setAlignment(Qt.AlignRight)



        lbl_layout = QHBoxLayout()
        lbl_layout.setAlignment(Qt.AlignLeft)
        lbl_layout.addWidget(self.width_lbl)
        lbl_layout.addWidget(self.height_lbl)



        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        
        main_layout.addLayout(lbl_layout)
        main_layout.addLayout(width_layout)
        main_layout.addWidget(self.ratio_check)
        main_layout.addLayout(apply_layout)

        self.setLayout(main_layout)

    def set_boxes(self):
        self.width_box.setText(str(_img_original.width))
        self.height_box.setText(str(_img_original.height))

    def on_width_change(self, e):
        logger.debug(f"type width {self.width_box.text()}")

        if self.ratio_check.isChecked():
            r_height = _get_ratio_height(_img_original.width, _img_original.height, int(self.width_box.text()))
            self.height_box.setText(str(r_height))

    def on_height_change(self, e):
        logger.debug(f"type height {self.height_box.text()}")

        if self.ratio_check.isChecked():
            r_width = _get_ratio_width(_img_original.width, _img_original.height, int(self.height_box.text()))
            self.width_box.setText(str(r_width))

    def on_ratio_change(self, e):
        logger.debug("ratio change")

    def on_apply(self, e):
        logger.debug("apply")

        operations.size = int(self.width_box.text()), int(self.height_box.text())

        self.parent.parent.update_img_size_lbl()

class AdjustingTab(QWidget):
    """Adjusting tab widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        contrast_lbl = QLabel("Contrast")
        contrast_lbl.setAlignment(Qt.AlignCenter)

        brightness_lbl = QLabel("Brightness")
        brightness_lbl.setAlignment(Qt.AlignCenter)

        sharpness_lbl = QLabel("Sharpness")
        sharpness_lbl.setAlignment(Qt.AlignCenter)

        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.setMinimum(SLIDER_MIN_VAL)
        self.contrast_slider.setMaximum(SLIDER_MAX_VAL)
        self.contrast_slider.sliderReleased.connect(self.on_contrast_slider_released)
        self.contrast_slider.setToolTip(str(SLIDER_MAX_VAL))

        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.setMinimum(SLIDER_MIN_VAL)
        self.brightness_slider.setMaximum(SLIDER_MAX_VAL)
        self.brightness_slider.sliderReleased.connect(self.on_brightness_slider_released)
        self.brightness_slider.setToolTip(str(SLIDER_MAX_VAL))

        self.sharpness_slider = QSlider(Qt.Horizontal, self)
        self.sharpness_slider.setMinimum(SLIDER_MIN_VAL)
        self.sharpness_slider.setMaximum(SLIDER_MAX_VAL)
        self.sharpness_slider.sliderReleased.connect(self.on_sharpness_slider_released)
        self.sharpness_slider.setToolTip(str(SLIDER_MAX_VAL))

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(contrast_lbl)
        main_layout.addWidget(self.contrast_slider)

        main_layout.addWidget(brightness_lbl)
        main_layout.addWidget(self.brightness_slider)

        main_layout.addWidget(sharpness_lbl)
        main_layout.addWidget(self.sharpness_slider)

        self.reset_sliders()
        self.setLayout(main_layout)

    def reset_sliders(self):
        self.brightness_slider.setValue(SLIDER_DEF_VAL)
        self.sharpness_slider.setValue(SLIDER_DEF_VAL)
        self.contrast_slider.setValue(SLIDER_DEF_VAL)

    def on_brightness_slider_released(self):
        logger.debug(f"brightness selected value: {self.brightness_slider.value()}")

        self.brightness_slider.setToolTip(str(self.brightness_slider.value()))

        factor = _get_converted_point(SLIDER_MIN_VAL, SLIDER_MAX_VAL, img_helper.BRIGHTNESS_FACTOR_MIN,
                                      img_helper.BRIGHTNESS_FACTOR_MAX, self.brightness_slider.value())
        logger.debug(f"brightness factor: {factor}")

        operations.brightness = factor

        self.parent.parent.place_preview_img()

    def on_sharpness_slider_released(self):
        logger.debug(self.sharpness_slider.value())

        self.sharpness_slider.setToolTip(str(self.sharpness_slider.value()))

        factor = _get_converted_point(SLIDER_MIN_VAL, SLIDER_MAX_VAL, img_helper.SHARPNESS_FACTOR_MIN,
                                      img_helper.SHARPNESS_FACTOR_MAX, self.sharpness_slider.value())
        logger.debug(f"sharpness factor: {factor}")

        operations.sharpness = factor

        self.parent.parent.place_preview_img()

    def on_contrast_slider_released(self):
        logger.debug(self.contrast_slider.value())

        self.contrast_slider.setToolTip(str(self.contrast_slider.value()))

        factor = _get_converted_point(SLIDER_MIN_VAL, SLIDER_MAX_VAL, img_helper.CONTRAST_FACTOR_MIN,
                                      img_helper.CONTRAST_FACTOR_MAX, self.contrast_slider.value())
        logger.debug(f"contrast factor: {factor}")

        operations.contrast = factor

        self.parent.parent.place_preview_img()


class FiltersTab(QWidget):
    """Color filters widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.main_layout = QHBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        self.add_filter_thumb("none")
        for key, val in color_filter.ColorFilters.filters.items():
            self.add_filter_thumb(key, val)

        self.setLayout(self.main_layout)

    def add_filter_thumb(self, name, title=""):
        logger.debug(f"create lbl thumb for: {name}")

        thumb_lbl = QLabel()
        thumb_lbl.name = name
        thumb_lbl.setStyleSheet("border:2px solid #ccc;")

        if name != "none":
            thumb_lbl.setToolTip(f"Apply <b>{title}</b> filter")
        else:
            thumb_lbl.setToolTip('No filter')

        thumb_lbl.setCursor(Qt.PointingHandCursor)
        thumb_lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb_lbl.mousePressEvent = partial(self.on_filter_select, name)

        self.main_layout.addWidget(thumb_lbl)

    def on_filter_select(self, filter_name, e):
        logger.debug(f"apply color filter: {filter_name}")
        global _img_preview
        operations.color_filter = filter_name
        self.toggle_thumbs()
        self.parent.parent.place_preview_img()

    def toggle_thumbs(self):
        for thumb in self.findChildren(QLabel):
            color = THUMB_BORDER_COLOR_ACTIVE if thumb.name == operations.color_filter else THUMB_BORDER_COLOR
            thumb.setStyleSheet(f"border:2px solid {color};")

class PaintableLabel(QLabel):
    def __init__(self, parent):
        super(PaintableLabel, self).__init__()
        self.setMouseTracking(True)
        self.ok_to_paint = False
        self.parent = parent
        self.PAINT_SIZE = 3

    def paint(self, img_np, x, y):  
        w = _img_preview.width
        h = _img_preview.height
        x = int(x * w/IMG_DIS_W)
        y = int(y * h/IMG_DIS_H)
        self.PAINT_SIZE = int(self.parent.pen_size* ( w/IMG_DIS_W + h/IMG_DIS_H )/2)
        if x > self.PAINT_SIZE and x < img_np.shape[1] -self.PAINT_SIZE and y > self.PAINT_SIZE and y < img_np.shape[0] - self.PAINT_SIZE: 
            img_np[y-self.PAINT_SIZE:y+self.PAINT_SIZE, x-self.PAINT_SIZE:x+self.PAINT_SIZE,:] = self.parent.pen_color 
       	return img_np
    
    def mouseMoveEvent(self, event):
        if self.ok_to_paint:
            global _img_preview
            img = _img_preview
            img = np.array(img)
            img = self.paint(img, event.pos().x(), event.pos().y())
            _img_preview = Image.fromarray(np.uint8(img))
            self.parent.place_preview_img()

    def mousePressEvent(self, event):
        print(event.pos().x(), event.pos().y())
        if self.parent.drawable:
            self.ok_to_paint = True    
	
    def mouseReleaseEvent(self, event):
        self.ok_to_paint = False

class MainLayout(QVBoxLayout):
    """Main layout"""

    def __init__(self, parent):
        super().__init__()  
        self.parent = parent
        self.drawable = False
        self.pen_color = 0
        self.pen_size = 3
        self.img_lbl = PaintableLabel(self)

        self.img_lbl.setFixedWidth(IMG_DIS_W)
        self.img_lbl.setFixedHeight(IMG_DIS_H)
        self.img_lbl.setText("<b>Image Editor @ ACH2 <b>"
                              "<div style='margin: 30px 0'><img src='HKUST.png' /></div>"
                              "<b>GUI Credits: ZENG Kuang, CHEN Liang-yu, WANG WenLong</b>")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.file_name = None

        self.img_size_lbl = None
        self.img_size_lbl = QLabel()
        self.img_size_lbl.setAlignment(Qt.AlignCenter)

        upload_btn = QPushButton("Upload")
        upload_btn.setMinimumWidth(BTN_MIN_WIDTH)
        upload_btn.clicked.connect(self.on_upload)
        upload_btn.setStyleSheet("font-weight:bold;")

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setMinimumWidth(BTN_MIN_WIDTH)
        self.reset_btn.clicked.connect(self.on_reset)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setStyleSheet("font-weight:bold;")

        self.save_btn = QPushButton("Save")
        self.save_btn.setMinimumWidth(BTN_MIN_WIDTH)
        self.save_btn.clicked.connect(self.on_save)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("font-weight:bold;")

        self.showOriginal_btn = QPushButton("original")
        self.showOriginal_btn.setMinimumWidth(BTN_MIN_WIDTH-20)
        self.showOriginal_btn.clicked.connect(self.on_showOriginal)
        self.showOriginal_btn.setEnabled(False)
        self.showOriginal_btn.setStyleSheet("font-weight:bold;")

        self.showProcessed_btn = QPushButton("processed")
        self.showProcessed_btn.setMinimumWidth(BTN_MIN_WIDTH-20)
        self.showProcessed_btn.clicked.connect(self.on_showProcessed)
        self.showProcessed_btn.setEnabled(False)
        self.showProcessed_btn.setStyleSheet("font-weight:bold;")

        self.addWidget(self.img_lbl)
        self.addWidget(self.img_size_lbl)
        self.addStretch()

        self.action_tabs = ActionTabs(self)
        self.addWidget(self.action_tabs)
        self.action_tabs.setVisible(False)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(upload_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.showOriginal_btn)
        btn_layout.addWidget(self.showProcessed_btn)

        self.addLayout(btn_layout)

    def place_preview_img(self):

        if not (operations.color_filter == None or operations.color_filter == 'none'):
            img = _get_img_with_all_operations()
            img = img_helper.color_filter(img, operations.color_filter)
        else:
            img = _get_img_with_all_operations()


        preview_pix = ImageQt.toqpixmap(img)
        self.img_lbl.setPixmap(preview_pix)

    def place_original_img(self):
        img = _img_original
        original_pix = ImageQt.toqpixmap(img)
        self.img_lbl.setPixmap(original_pix)
    

    def on_save(self):
        logger.debug("open save dialog")
        new_img_path, _ = QFileDialog.getSaveFileName(self.parent, "QFileDialog.getSaveFileName()",
                                                      f"ez_pz_{self.file_name}",
                                                      "Images (*.png *.jpg)")

        if new_img_path:
            logger.debug(f"save output image to {new_img_path}")
            if not (operations.color_filter == None or operations.color_filter == 'none'):
                img = _get_img_with_all_operations()
                img = img_helper.color_filter(img, operations.color_filter)
            else:
                img = _get_img_with_all_operations()
            
            img.save(new_img_path)

    def on_upload(self):
        logger.debug("upload")
        img_path, _ = QFileDialog.getOpenFileName(self.parent, "Open image",
                                                  "/Users",
                                                  "Images (*.png *jpg)")

        if img_path:
            logger.debug(f"open file {img_path}")

            self.file_name = ntpath.basename(img_path)

            pix = QPixmap(img_path)
            self.img_lbl.setPixmap(pix)
            self.action_tabs.setVisible(True)
            self.action_tabs.adjustment_tab.reset_sliders()
            self.img_lbl.setScaledContents(True)

            global _img_original
            _img_original = ImageQt.fromqpixmap(pix)

            self.update_img_size_lbl()

            if _img_original.width < _img_original.height:
                w = THUMB_SIZE
                h = _get_ratio_height(_img_original.width, _img_original.height, w)
            else:
                h = THUMB_SIZE
                w = _get_ratio_width(_img_original.width, _img_original.height, h)
			
            # if _img_original.width > self.img_lbl.width() or _img_original.height > self.img_lbl.height():
            #    self.img_lbl.setScaledContents(True)
		
            img_filter_thumb = img_helper.resize(_img_original, w, h)

            global _img_preview
            _img_preview = _img_original.copy()

            for thumb in self.action_tabs.filters_tab.findChildren(QLabel):
                if thumb.name != "none":
                    img_filter_preview = img_helper.color_filter(img_filter_thumb, thumb.name)
                else:
                    img_filter_preview = img_filter_thumb

                preview_pix = ImageQt.toqpixmap(img_filter_preview)
                thumb.setPixmap(preview_pix)

            self.reset_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.showOriginal_btn.setEnabled(True)
            self.action_tabs.modification_tab.set_boxes()
            global win
            win.setWindowTitle('ACH2 FYP process')

    def update_img_size_lbl(self):
        logger.debug("update img size lbl")

        self.img_size_lbl.setText(f"<span style='font-size:11px'>"
                                  f"image size {operations.size[0] if operations.size else _img_original.width} × "
                                  f"{operations.size[1] if operations.size else _img_original.height}"
                                  f"</span>")

    def on_reset(self):
        logger.debug("reset all")

        global _img_preview
        _img_preview = _img_original.copy()
        global _bimap
        _bimap = None

        operations.reset()

        self.action_tabs.filters_tab.toggle_thumbs()
        self.place_preview_img()
        self.action_tabs.adjustment_tab.reset_sliders()
        self.action_tabs.modification_tab.set_boxes() 
        self.action_tabs.inpainting_tab.exemplar_btn.setEnabled(True)
        self.action_tabs.inpainting_tab.ref_btn.setEnabled(True)
        self.action_tabs.seg_tab.seg_btn.setEnabled(True)
        self.action_tabs.inpainting_tab.setEnabled(False)
        self.update_img_size_lbl()


    def on_showOriginal(self):
        logger.debug("show Original image")
        self.showProcessed_btn.setEnabled(True)
        self.showOriginal_btn.setEnabled(False)
        self.place_original_img()
        self.update_img_size_lbl()

    def on_showProcessed(self):
        logger.debug("show processed image")
        self.showProcessed_btn.setEnabled(False)
        self.showOriginal_btn.setEnabled(True)
        self.place_preview_img()
        self.update_img_size_lbl()
        


class EasyPzUI(QWidget):
    """Main widget"""

    def __init__(self):
        super().__init__()

        self.main_layout = MainLayout(self)
        self.setLayout(self.main_layout)

        self.setMinimumSize(600, 500)
        self.setMaximumSize(900, 900)
        self.setGeometry(600, 600, 600, 600)
        self.setWindowTitle('ACH2 FYP HOME')
        self.center()
        self.show()

    def center(self):
        """align window center"""

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        logger.debug("close")

        if operations.has_changes():
            reply = QMessageBox.question(self, "",
                                         "You have unsaved changes<br>Are you sure?", QMessageBox.Yes |
                                         QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

    def resizeEvent(self, e):
        pass


class QVLine(QFrame):
    """Vertical line"""

    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


if __name__ == '__main__':
    fileConfig('logging_config.ini')

    app = QApplication(sys.argv)
    win = EasyPzUI()
    sys.exit(app.exec_())


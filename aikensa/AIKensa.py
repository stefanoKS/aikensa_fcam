import cv2
import sys
import yaml
import os
from enum import Enum
import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
# from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.cam_thread import CameraThread, CameraConfig
from aikensa.calibration_thread import CalibrationThread, CalibrationConfig

from aikensa.sio_thread import ServerMonitorThread
from aikensa.time_thread import TimeMonitorThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',             # index 0
    'aikensa/qtui/calibration_cam1.ui', # index 1
    'aikensa/qtui/calibration_cam2.ui', # index 2
    'aikensa/qtui/calibration_cam3.ui', # index 3
    'aikensa/qtui/calibration_cam4.ui', # index 4
    'aikensa/qtui/calibration_cam5.ui', # index 5
    'aikensa/qtui/edgedetection.ui',        # index 6
    "aikensa/qtui/65820W030P.ui",             # index 7
    "aikensa/qtui/empty.ui", #empty 8
    "aikensa/qtui/empty.ui", #empty 9
    "aikensa/qtui/empty.ui", #empty 10
    "aikensa/qtui/empty.ui", #empty 11
    "aikensa/qtui/empty.ui", #empty 12
    "aikensa/qtui/empty.ui", #empty 13
    "aikensa/qtui/empty.ui", #empty 14
    "aikensa/qtui/empty.ui", #empty 15
    "aikensa/qtui/empty.ui", #empty 16
    "aikensa/qtui/empty.ui", #empty 17
    "aikensa/qtui/empty.ui", #empty 18
    "aikensa/qtui/empty.ui", #empty 19
    "aikensa/qtui/empty.ui", #empty 20
    "aikensa/qtui/dailyTenken4go_01.ui",  # index 21
    "aikensa/qtui/dailyTenken4go_02.ui",  # index 22
    "aikensa/qtui/dailyTenken4go_03.ui",  # index 23
]


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # self.cam_thread = CameraThread(CameraConfig())
        self.calibration_thread = CalibrationThread(CalibrationConfig())
        self._setup_ui()
        # self.cam_thread.start()
        self.calibration_thread.start()

        # Thread for SiO
        HOST = '192.168.0.100'  # Use the IP address from SiO settings
        PORT = 40001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.08)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.start()

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.initial_colors = {}#store initial colors of the labels

    def timeUpdate(self, time):
        for label in self.timeLabel:
            label.setText(time)

    def handle_server_status(self, is_up):
        status_text = "ON" if is_up else "OFF"
        status_color = "green" if is_up else "red"

        #to show the label later. Implement later

        for label in self.siostatus_server:
            if label:  # Check if the label was found correctly
                label.setText(status_text)
                label.setStyleSheet(f"color: {status_color};")


    def handle_input_states(self, input_states):
        # check if input_stats is not empty
        if input_states:
            if input_states[0] == 1:
                self.trigger_kensa()
            if input_states[1] == 1:
                self.trigger_rekensa()
            else:
                pass

    def trigger_kensa(self):
        self.button_kensa3.click()
        self.button_kensa4.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):

        self.calibration_thread.CalibCamStream.connect(self._setCalibFrame)


        # self.cam_thread.camFrame1.connect(self._setFrameCam1)
        # self.cam_thread.camFrame2.connect(self._setFrameCam2)
        # self.cam_thread.mergeFrame.connect(self._setFrameMerge)
        
        # self.cam_thread.kata1Frame.connect(self._setFrameKata1)
        # self.cam_thread.kata2Frame.connect(self._setFrameKata2)
        
        # self.cam_thread.clip1Frame.connect(self._setFrameClip1)
        # self.cam_thread.clip2Frame.connect(self._setFrameClip2)
        # self.cam_thread.clip3Frame.connect(self._setFrameClip3)

        # self.cam_thread.handFrame1.connect(lambda paramValue: self._set_labelFrame(self.stackedWidget.widget(3), paramValue, "clip1Check"))
        # self.cam_thread.handFrame2.connect(lambda paramValue: self._set_labelFrame(self.stackedWidget.widget(3), paramValue, "clip2Check"))
        # self.cam_thread.handFrame3.connect(lambda paramValue: self._set_labelFrame(self.stackedWidget.widget(3), paramValue, "clip3Check"))
        # self.cam_thread.handFrame1.connect(lambda paramValue: self._set_labelFrame(self.stackedWidget.widget(4), paramValue, "clip1Check"))
        # self.cam_thread.handFrame2.connect(lambda paramValue: self._set_labelFrame(self.stackedWidget.widget(4), paramValue, "clip2Check"))
        # self.cam_thread.handFrame3.connect(lambda paramValue: self._set_labelFrame(self.stackedWidget.widget(4), paramValue, "clip3Check"))

        # self.cam_thread.ctrplrworkorderSignal.connect(self._set_workorder_color_ctrplr)

        # self.cam_thread.ctrplrLH_pitch_updated.connect(self._set_button_color_ctrplr)
        # self.cam_thread.ctrplrRH_pitch_updated.connect(self._set_button_color_ctrplr)

        # self.cam_thread.ctrplrLH_currentnumofPart_updated.connect(self._set_numlabel_text_ctrplr_LH_current)
        # self.cam_thread.ctrplrRH_currentnumofPart_updated.connect(self._set_numlabel_text_ctrplr_RH_current)

        # self.cam_thread.ctrplrLH_numofPart_updated.connect(self._set_numlabel_text_ctrplr_LH_total)
        # self.cam_thread.ctrplrRH_numofPart_updated.connect(self._set_numlabel_text_ctrplr_RH_total)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        cameraCalibration1_widget = self.stackedWidget.widget(1)
        cameraCalibration2_widget = self.stackedWidget.widget(2)
        cameraCalibration3_widget = self.stackedWidget.widget(3)
        cameraCalibration4_widget = self.stackedWidget.widget(4)
        cameraCalibration5_widget = self.stackedWidget.widget(5)
        cameraCalibration1_button = main_widget.findChild(QPushButton, "camcalibrationbutton1")
        cameraCalibration2_button = main_widget.findChild(QPushButton, "camcalibrationbutton2")
        cameraCalibration3_button = main_widget.findChild(QPushButton, "camcalibrationbutton3")
        cameraCalibration4_button = main_widget.findChild(QPushButton, "camcalibrationbutton4")
        cameraCalibration5_button = main_widget.findChild(QPushButton, "camcalibrationbutton5")

        edgedetection_widget = self.stackedWidget.widget(6)
        edgedetection_button = main_widget.findChild(QPushButton, "edgedetectbutton")

        dailytenken01_widget = self.stackedWidget.widget(21)
        dailytenken02_widget = self.stackedWidget.widget(22)
        dailytenken03_widget = self.stackedWidget.widget(23)
        button_dailytenken01 = main_widget.findChild(QPushButton, "dailytenkenbutton")
        button_dailytenken02 = dailytenken01_widget.findChild(QPushButton, "nextButton")
        button_dailytenken03 = dailytenken02_widget.findChild(QPushButton, "nextButton")
        button_dailytenken_kanryou = dailytenken03_widget.findChild(QPushButton, "finishButton")

        self.siostatus = main_widget.findChild(QLabel, "status_sio")
        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in [0, 7]]

        if cameraCalibration1_button:
            cameraCalibration1_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            cameraCalibration1_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 1))

        if cameraCalibration2_button:
            cameraCalibration2_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
            cameraCalibration2_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 2))

        if cameraCalibration3_button:
            cameraCalibration3_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
            cameraCalibration3_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 3))

        if cameraCalibration4_button:
            cameraCalibration4_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
            cameraCalibration4_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 4))

        if cameraCalibration5_button:
            cameraCalibration5_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(5))
            cameraCalibration5_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 5))

        # if button_calib:
        #     button_calib.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        #     button_calib.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 1))

        # if edgedetection_button:
        #     edgedetection_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        #     edgedetection_button.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 2))

        # if button_P5755A491:
        #     button_P5755A491.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        #     button_P5755A491.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 3))

        # if button_P5755A492:
        #     button_P5755A492.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
        #     button_P5755A492.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 4))

        # if button_dailytenken01:
        #     button_dailytenken01.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        #     button_dailytenken01.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 21))

        # if button_dailytenken02:
        #     button_dailytenken02.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        #     button_dailytenken02.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 22))

        # if button_dailytenken03:
        #     button_dailytenken03.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(23))
        #     button_dailytenken03.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 23))


        # add extra widgets here

        # Widget 0 -> <Main Page>

        
        #Widget 1 # Calibration and Sample
        # captureButton1 = self.stackedWidget.widget(1).findChild(QPushButton, "takeImage1")
        # captureButton2 = self.stackedWidget.widget(1).findChild(QPushButton, "takeImage2")
        # captureButton1.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "captureCam1", True))
        # captureButton2.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "captureCam2", True))

        # self.connect_camparam_button(1, "takeImageClip1", "captureClip1", True)
        # self.connect_camparam_button(1, "takeImageClip2", "captureClip2", True)
        # self.connect_camparam_button(1, "takeImageClip3", "captureClip3", True)
        

        # cam1CalibrateButton = self.stackedWidget.widget(1).findChild(QPushButton, "calibCam1")
        # cam1CalibrateButton.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "calculateCamMatrix1", True))

        # cam2CalibrateButton = self.stackedWidget.widget(1).findChild(QPushButton, "calibCam2")
        # cam2CalibrateButton.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "calculateCamMatrix2", True))



        # Widget 2
        # button_saveparam = self.stackedWidget.widget(2).findChild(QPushButton, "saveparambutton")
        # button_saveparam.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "savecannyparams", True))

        # button_takecanny = self.stackedWidget.widget(2).findChild(QPushButton, "takeimagebutton")
        # button_takecanny.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "capture", True))

        # button_readwarp = self.stackedWidget.widget(2).findChild(QPushButton, "button_readwarp")
        # label_readwarp = self.stackedWidget.widget(2).findChild(QLabel, "label_readwarpcolor")
        # button_readwarp.pressed.connect(lambda: self._toggle_param_and_update_label("cannyreadwarp", label_readwarp))

        

        # # frame = process_for_edge_detection(frame, self.slider_value)
        # slider_opacity = self.stackedWidget.widget(2).findChild(QSlider, "slider_opacity")
        # slider_blur = self.stackedWidget.widget(2).findChild(QSlider, "slider_blur")
        # slider_lowercanny = self.stackedWidget.widget(2).findChild(QSlider, "slider_lowercanny")
        # slider_uppercanny = self.stackedWidget.widget(2).findChild(QSlider, "slider_uppercanny")
        # slider_contrast = self.stackedWidget.widget(2).findChild(QSlider, "slider_contrast")
        # slider_brightness = self.stackedWidget.widget(2).findChild(QSlider, "slider_brightness")

        # slider_opacity.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'opacity', x/100))
        # slider_blur.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'blur', x))
        # slider_lowercanny.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'lower_canny', x))
        # slider_uppercanny.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'upper_canny', x))
        # slider_contrast.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'contrast', x/100))
        # slider_brightness.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'brightness', x/100))

        ## Widget 3 and 4

        # button_HDRes = self.connect_button_font_color_change(3, "button_HDResQT", "HDRes")
        
        # kensaButton = self.stackedWidget.widget(3).findChild(QPushButton, "kensaButton")
        # kensaButton.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))

        # button_HDRes4 = self.connect_button_font_color_change(4, "button_HDResQT", "HDRes")
        
        # kensaButton4 = self.stackedWidget.widget(4).findChild(QPushButton, "kensaButton")
        # kensaButton4.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))

        # self.connect_camparam_button(3, "counterReset", "resetCounter", True)
        # self.connect_camparam_button(4, "counterReset", "resetCounter", True)

        # self.connect_line_edit_text_changed(widget_index=3, line_edit_name="kensain_name", cam_param="kensainName")
        # self.connect_line_edit_text_changed(widget_index=4, line_edit_name="kensain_name", cam_param="kensainName")

        # kensaresetButton = self.stackedWidget.widget(3).findChild(QPushButton, "kensareset")
        # kensaresetButton.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "kensaReset", True))
        
        # kensaresetButton4 = self.stackedWidget.widget(4).findChild(QPushButton, "kensareset")
        # kensaresetButton4.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "kensaReset", True))

        # workorder1 = self.stackedWidget.widget(3).findChild(QLineEdit, "order1")
        # workorder2 = self.stackedWidget.widget(3).findChild(QLineEdit, "order2")
        # workorder3 = self.stackedWidget.widget(3).findChild(QLineEdit, "order3")

        # self.button_kensa3 = self.stackedWidget.widget(3).findChild(QPushButton, "kensaButton")
        # self.button_kensa4 = self.stackedWidget.widget(4).findChild(QPushButton, "kensaButton")

        self.siostatus_server = [self.stackedWidget.widget(i).findChild(QLabel, "status_sio") for i in [0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23]]

        # self.kanseihin_number_ctrplr_lh = self.stackedWidget.widget(3).findChild(QLabel, "status_kansei")
        # self.furyouhin_number_ctrplr_lh = self.stackedWidget.widget(3).findChild(QLabel, "status_furyou")
        # self.kanseihin_number_current_ctrplr_lh = self.stackedWidget.widget(3).findChild(QLabel, "current_kansei")
        # self.furyouhin_number_current_ctrplr_lh = self.stackedWidget.widget(3).findChild(QLabel, "current_furyou")

        # self.kanseihin_number_ctrplr_rh = self.stackedWidget.widget(4).findChild(QLabel, "status_kansei")
        # self.furyouhin_number_ctrplr_rh = self.stackedWidget.widget(4).findChild(QLabel, "status_furyou")
        # self.kanseihin_number_current_ctrplr_rh = self.stackedWidget.widget(4).findChild(QLabel, "current_kansei")
        # self.furyouhin_number_current_ctrplr_rh = self.stackedWidget.widget(4).findChild(QLabel, "current_furyou")

        # for i in [3, 4]:
        #     self.connect_camparam_button(i, "kansei_plus", "kansei_plus", True)
        #     self.connect_camparam_button(i, "kansei_minus", "kansei_minus", True)
        #     self.connect_camparam_button(i, "furyou_plus", "furyou_plus", True)
        #     self.connect_camparam_button(i, "furyou_minus", "furyou_minus", True)
        #     self.connect_camparam_button(i, "kansei_plus_10", "kansei_plus_10", True)
        #     self.connect_camparam_button(i, "kansei_minus_10", "kansei_minus_10", True)
        #     self.connect_camparam_button(i, "furyou_plus_10", "furyou_plus_10", True)
        #     self.connect_camparam_button(i, "furyou_minus_10", "furyou_minus_10", True)


                #kensabutton for dailytenken
        # self.button_dailyTenken01 = self.stackedWidget.widget(21).findChild(QPushButton, "checkButton")
        # self.button_dailyTenken02 = self.stackedWidget.widget(22).findChild(QPushButton, "checkButton")
        # self.button_dailyTenken03 = self.stackedWidget.widget(23).findChild(QPushButton, "checkButton")

        # self.button_dailyTenken01.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "kensaButton", True))
        # self.button_dailyTenken02.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "kensaButton", True))
        # self.button_dailyTenken03.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "kensaButton", True))

        # self.button_dailyTenken01.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))
        # self.button_dailyTenken02.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))
        # self.button_dailyTenken03.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))

       # Find and connect quit buttons and main menu buttons in all widgets
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                # button_main_menu.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 0))
                button_main_menu.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 0))
                button_dailytenken_kanryou.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                # button_dailytenken_kanryou.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 0))



        self.stackedWidget.currentChanged.connect(self._on_widget_changed)

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()



    def connect_button_font_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: black")
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)
                new_color = "red" if new_value else "black"
                button.setStyleSheet(f"color: {new_color}")
            button.pressed.connect(toggle_font_color_and_param)
        else:
            print(f"Button '{qtbutton}' not found.")

    def connect_button_label_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: red")
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)
                new_color = "green" if new_value else "red"
                button.setStyleSheet(f"color: {new_color}")

            button.pressed.connect(toggle_font_color_and_param)
        else:
            print(f"Button '{qtbutton}' not found.")

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_cam_params(self.cam_thread, cam_param, text))

    def connect_camparam_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_cam_params(self.cam_thread, cam_param, value))
            print(f"Button '{button_name}' connected to cam_param '{cam_param}' with value '{value}' in widget {widget_index}")

    def simulateButtonKensaClicks(self):
        self.button_kensa3.click()
        self.button_kensa4.click()

    def _on_widget_changed(self, idx: int):
        if idx in [7, 21, 22, 23]:
            #Change widget value to equal to index of stacked widget first
            self._set_cam_params(self.cam_thread, 'widget', idx)
            self.cam_thread.initialize_model()
            

    def _close_app(self):
        # self.cam_thread.stop()
        self.calibration_thread.stop()
        self.server_monitor_thread.stop()
        time.sleep(1.0)
        QCoreApplication.instance().quit()

    def _load_ui(self, filename):
        widget = QMainWindow()
        loadUi(filename, widget)
        return widget

    def _set_frame_raw(self, image):
        for i in [1, 2]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_frame_inference(self, image):
        for i in [3, 4]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_cam_params(self, thread, key, value):
        setattr(thread.cam_config, key, value)

    def _toggle_param_and_update_label(self, param, label):
        # Toggle the parameter value
        new_value = not getattr(self.cam_thread.cam_config, param)
        self._set_cam_params(self.cam_thread, param, new_value)

        # Update the label color based on the new parameter value
        color = "green" if new_value else "red"
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_labelFrame(self, widget, paramValue, label_names):
        colorOK = "blue"
        colorNG = "black"
        label = widget.findChild(QLabel, label_names) 
        color = colorNG if paramValue else colorOK
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")
        
    def _set_button_color(self, pitch_data):
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    # def _set_numlabel_text_ctrplr_LH_current(self, numofPart):
    #     self.kanseihin_number_current_ctrplr_lh.setText(str(numofPart[0]))
    #     self.furyouhin_number_current_ctrplr_lh.setText(str(numofPart[1]))

    # def _set_numlabel_text_ctrplr_RH_current(self, numofPart):
    #     self.kanseihin_number_current_ctrplr_rh.setText(str(numofPart[0]))
    #     self.furyouhin_number_current_ctrplr_rh.setText(str(numofPart[1]))

    # def _set_numlabel_text_ctrplr_LH_total(self, numofPart):
    #     self.kanseihin_number_ctrplr_lh.setText(str(numofPart[0]))
    #     self.furyouhin_number_ctrplr_lh.setText(str(numofPart[1]))

    # def _set_numlabel_text_ctrplr_RH_total(self, numofPart):
    #     self.kanseihin_number_ctrplr_rh.setText(str(numofPart[0]))
    #     self.furyouhin_number_ctrplr_rh.setText(str(numofPart[1]))
    

    def _setCalibFrame(self, image):
        for i in [1, 2, 3, 4, 5]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camFrame")
            label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameCam1(self, image):
    #     widget = self.stackedWidget.widget(1)
    #     label = widget.findChild(QLabel, "camFrame1")
    #     label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameCam2(self, image):
    #     widget = self.stackedWidget.widget(1)
    #     label = widget.findChild(QLabel, "camFrame2") 
    #     label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameMerge(self, image):
    #     for i in [1, 3, 4, 21, 22, 23]:
    #         widget = self.stackedWidget.widget(i)
    #         label = widget.findChild(QLabel, "mergeFrame")
    #         label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameKata1(self, image):
    #     for i in [1, 3, 4]:
    #         widget = self.stackedWidget.widget(i)
    #         label = widget.findChild(QLabel, "kata1Frame") 
    #         label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameKata2(self, image):
    #     for i in [1, 3, 4]:
    #         widget = self.stackedWidget.widget(i)
    #         label = widget.findChild(QLabel, "kata2Frame") 
    #         label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameClip1(self, image):
    #     for i in [1, 3, 4]: #modify this later
    #         widget = self.stackedWidget.widget(i)
    #         label = widget.findChild(QLabel, "clip1Frame") 
    #         label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameClip2(self, image):
    #     for i in [1, 3, 4]: #modify this later
    #         widget = self.stackedWidget.widget(i)
    #         label = widget.findChild(QLabel, "clip2Frame") 
    #         label.setPixmap(QPixmap.fromImage(image))

    # def _setFrameClip3(self, image):
    #     for i in [1, 3, 4]: #modify this later
    #         widget = self.stackedWidget.widget(i)
    #         label = widget.findChild(QLabel, "clip3Frame") 
    #         label.setPixmap(QPixmap.fromImage(image))

    def _extract_color(self, stylesheet):
        # Extracts the color value from the stylesheet string
        start = stylesheet.find("background-color: ") + len("background-color: ")
        end = stylesheet.find(";", start)
        return stylesheet[start:end].strip()

    def _store_initial_colors(self, widget_index, label_names):
        if widget_index not in self.initial_colors:
            self.initial_colors[widget_index] = {}
        labels = [self.stackedWidget.widget(widget_index).findChild(QLabel, name) for name in label_names]
        for label in labels:
            color = self._extract_color(label.styleSheet())
            self.initial_colors[widget_index][label.objectName()] = color
            # print(f"Stored initial color for {label.objectName()} in widget {widget_index}: {color}")

    def _set_workorder_color_ctrplr(self, workOrder): # For rr side, consists of 6 pitches and Lsun (total Length)
        colorOK = "green"
        colorNG = "red"
        label_names = ["order1", "order2", "order3", "order4", "order5"]

        if not self.initial_colors:
            for widget_index in [3, 4]:
                self._store_initial_colors(widget_index, label_names)

        for widget_index in [3, 4]:
            labels = [self.stackedWidget.widget(widget_index).findChild(QLabel, name) for name in label_names]
            
            for i, pitch_value in enumerate(workOrder):
                if pitch_value:
                    color = colorOK
                else:
                    color = self.initial_colors[widget_index][labels[i].objectName()] # Use the initial color from the specific widget
                labels[i].setStyleSheet(f"QLabel {{background-color: {color};border-radius: 13px;min-height: 10px;min-width: 10px;}}")
                # print(f"Setting color for {labels[i].objectName()} in widget {widget_index} to {color}")


    def _set_button_color_ctrplr(self, pitch_data): #For rr side, consists of 6 pitches and Lsun (total Length)
        colorOK = "green"
        colorNG = "red"
        # print (pitch_data)
        label_names = ["P1color", "P2color", "P3color",
                        "P4color", "P5color", "P6color",
                        "P7color", "P8color"]
        
        for widget_index in [3, 4]:
            labels = [self.stackedWidget.widget(widget_index).findChild(QLabel, name) for name in label_names]
            
            for i, pitch_value in enumerate(pitch_data):
                if i >= len(labels):
                    break #in case the number of pitches is more than the number of labels
                color = colorOK if pitch_value else colorNG
                labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

            
    def _set_calib_params(self, thread, key, value):
        setattr(thread.calib_config, key, value)


def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
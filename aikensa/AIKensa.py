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
from aikensa.inspection_thread import InspectionThread, InspectionConfig

from aikensa.sio_thread import ServerMonitorThread, ServerConfig
from aikensa.time_thread import TimeMonitorThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',             # index 0
    'aikensa/qtui/calibration_cam1.ui', # index 1
    'aikensa/qtui/calibration_cam2.ui', # index 2
    'aikensa/qtui/calibration_cam3.ui', # index 3
    'aikensa/qtui/calibration_cam4.ui', # index 4
    'aikensa/qtui/calibration_cam5.ui', # index 5
    'aikensa/qtui/camera_merge.ui',                # index 6
    'aikensa/qtui/edgedetection.ui',        # index 7
    "aikensa/qtui/P65820W030P.ui",           # Mitsubishi 5A45 Hood FR index 8
    "aikensa/qtui/P658207YA0A.ui" ,          # Nissan Hood FR index 9
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
        self.inspection_thread = InspectionThread(InspectionConfig())   
        self._setup_ui()
        # self.cam_thread.start()
        # self.calibration_thread.start()
        # self.inspection_thread.start()

        # Thread for SiO
        HOST = '192.168.0.100'  # Use the IP address from SiO settings
        PORT = 30001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.1)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.input_states_signal.connect(self._sentInputToInspectionThread)
        
        self.server_monitor_thread.start()

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.initial_colors = {}#store initial colors of the labels

        self.widget_dir_map = {
            8: "65820W030P",
            9: "658207YA0A",
        }

        self.prevTriggerStates = 0
        self.TriggerWaitTime = 2.0
        self.currentTime = time.time()

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
        # print(f"Input states: {input_states}")
        if input_states:
            if input_states[5] == 1 and self.prevTriggerStates == 0:
                self.trigger_kensa()
                self.prevTriggerStates = input_states[5]
                # print("Triggered Kensa")
            if time.time() - self.currentTime > self.TriggerWaitTime:
                # print("timePassed")
                self.prevTriggerStates = 0
                self.currentTime = time.time()
            else:
                pass

    def trigger_kensa(self):
        self.Inspect_button.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):

        self.calibration_thread.CalibCamStream.connect(self._setCalibFrame)
        self.calibration_thread.CamMerge1.connect(self._setMergeFrame1)
        self.calibration_thread.CamMerge2.connect(self._setMergeFrame2)
        self.calibration_thread.CamMerge3.connect(self._setMergeFrame3)
        self.calibration_thread.CamMerge4.connect(self._setMergeFrame4)
        self.calibration_thread.CamMerge5.connect(self._setMergeFrame5)
        self.calibration_thread.CamMergeAll.connect(self._setMergeFrameAll)

        self.inspection_thread.part1Cam.connect(self._setPartFrame1)
        self.inspection_thread.part2Cam.connect(self._setPartFrame2)
        self.inspection_thread.part3Cam.connect(self._setPartFrame3)
        self.inspection_thread.part4Cam.connect(self._setPartFrame4)
        self.inspection_thread.part5Cam.connect(self._setPartFrame5)

        self.inspection_thread.hole1Cam.connect(self._setHoleFrame1)
        self.inspection_thread.hole2Cam.connect(self._setHoleFrame2)
        self.inspection_thread.hole3Cam.connect(self._setHoleFrame3)
        self.inspection_thread.hole4Cam.connect(self._setHoleFrame4)
        self.inspection_thread.hole5Cam.connect(self._setHoleFrame5)

        self.inspection_thread.dailytenkenCam.connect(self._dailyTenkenFrame)

        self.inspection_thread.hoodFR_InspectionResult_PitchMeasured.connect(self._outputMeasurementText)
        self.inspection_thread.hoodFR_InspectionResult_PitchResult.connect(self._outputMeasurementTextColor)
        self.inspection_thread.hoodFR_InspectionStatus.connect(self._inspectionStatusText)

        self.inspection_thread.hoodFR_HoleStatus.connect(self._inspectionStatusHole)

        self.inspection_thread.P658207YA0A_InspectionResult_PitchMeasured.connect(self._P658207YA0A_outputMeasurementText)
        self.inspection_thread.P658207YA0A_InspectionResult_PitchResult.connect(self._P658207YA0A_outputMeasurementTextColor)
        self.inspection_thread.P658207YA0A_InspectionStatus.connect(self._P658207YA0A_inspectionStatusText)


        self.inspection_thread.ethernet_status_red_tenmetsu.connect(self._setEthernetStatusTenmetsuRed)
        self.inspection_thread.ethernet_status_green_hold.connect(self._setEthernetStatusHoldGreen)
        self.inspection_thread.ethernet_status_red_hold.connect(self._setEthernetStatusHoldRed)
        self.inspection_thread.cropPositionChanged.connect(self._setCropValueLabels)

        self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)


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
        mergeCamera_widget = self.stackedWidget.widget(6)
        partInspection_P65820W030P = self.stackedWidget.widget(8)


        cameraCalibration1_button = main_widget.findChild(QPushButton, "camcalibrationbutton1")
        cameraCalibration2_button = main_widget.findChild(QPushButton, "camcalibrationbutton2")
        cameraCalibration3_button = main_widget.findChild(QPushButton, "camcalibrationbutton3")
        cameraCalibration4_button = main_widget.findChild(QPushButton, "camcalibrationbutton4")
        cameraCalibration5_button = main_widget.findChild(QPushButton, "camcalibrationbutton5")
        mergeCamera_button = main_widget.findChild(QPushButton, "cameraMerge")
        # partInspection_P65820W030P_button = main_widget.findChild(QPushButton, "P65820W030Pbutton")

        button_config = {
            "P65820W030Pbutton": {"widget_index": 8, "inspection_param": 8},
            "P658207LE0Abutton": {"widget_index": 9, "inspection_param": 9},
        }

        for button_name, config in button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                # Connect each signal with the necessary parameters
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
                button.clicked.connect(self.calibration_thread.stop)


        dailytenken01_P65820W030P_widget = self.stackedWidget.widget(21)
        dailytenken01_P65820W030P_button = main_widget.findChild(QPushButton, "dailytenkenbutton")
        dailytenken01_P65820W030P_kanryou_button = dailytenken01_P65820W030P_widget.findChild(QPushButton, "finishButton")

        self.siostatus = main_widget.findChild(QLabel, "status_sio")
        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        if cameraCalibration1_button:
            cameraCalibration1_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            cameraCalibration1_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 1))
            cameraCalibration1_button.clicked.connect(self.calibration_thread.start)
        if cameraCalibration2_button:
            cameraCalibration2_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
            cameraCalibration2_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 2))
            cameraCalibration2_button.clicked.connect(self.calibration_thread.start)
        if cameraCalibration3_button:
            cameraCalibration3_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
            cameraCalibration3_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 3))
            cameraCalibration3_button.clicked.connect(self.calibration_thread.start)
        if cameraCalibration4_button:
            cameraCalibration4_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
            cameraCalibration4_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 4))
            cameraCalibration4_button.clicked.connect(self.calibration_thread.start)
        if cameraCalibration5_button:
            cameraCalibration5_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(5))
            cameraCalibration5_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 5))
            cameraCalibration5_button.clicked.connect(self.calibration_thread.start)
        if mergeCamera_button:
            mergeCamera_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
            mergeCamera_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 6))
            mergeCamera_button.clicked.connect(self.calibration_thread.start)
        # if partInspection_P65820W030P_button:
        #     partInspection_P65820W030P_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(8))
        #     partInspection_P65820W030P_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 8))
        #     partInspection_P65820W030P_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        #     partInspection_P65820W030P_button.clicked.connect(self.calibration_thread.stop)
        if dailytenken01_P65820W030P_button:
            dailytenken01_P65820W030P_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
            dailytenken01_P65820W030P_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))
            dailytenken01_P65820W030P_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
            dailytenken01_P65820W030P_button.clicked.connect(self.calibration_thread.stop)        

        for i in range(1, 6):
            CalibrateSingleFrame = self.stackedWidget.widget(i).findChild(QPushButton, "calibSingleFrame")
            CalibrateSingleFrame.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateSingeFrameMatrix", True))

            CalibrateFinalCameraMatrix = self.stackedWidget.widget(i).findChild(QPushButton, "calibCam")
            CalibrateFinalCameraMatrix.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateCamMatrix", True))

        calcHomoCam1 = mergeCamera_widget.findChild(QPushButton, "calcH_cam1")
        calcHomoCam2 = mergeCamera_widget.findChild(QPushButton, "calcH_cam2")
        calcHomoCam3 = mergeCamera_widget.findChild(QPushButton, "calcH_cam3")
        calcHomoCam4 = mergeCamera_widget.findChild(QPushButton, "calcH_cam4")
        calcHomoCam5 = mergeCamera_widget.findChild(QPushButton, "calcH_cam5")

        calcHomoCam1.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam1", True))
        calcHomoCam2.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam2", True))
        calcHomoCam3.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam3", True))
        calcHomoCam4.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam4", True))
        calcHomoCam5.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam5", True))

        planarize_combined = mergeCamera_widget.findChild(QPushButton, "planarize")
        if planarize_combined is not None:
            planarize_combined.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize", True))

        planarize_temp_combined = mergeCamera_widget.findChild(QPushButton, "planarize_temp")
        if planarize_temp_combined is not None:
            planarize_temp_combined.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize_temp", True))

        planarize_j42u_combined = mergeCamera_widget.findChild(QPushButton, "planarize_J42U")
        if planarize_j42u_combined is not None:
            planarize_j42u_combined.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize_temp", True))

        for camera_index in range(1, 6):
            self.connect_camera_merge_adjustment_button(6, f"CAM{camera_index}_LEFT", camera_index, x_delta=-0.05)
            self.connect_camera_merge_adjustment_button(6, f"CAM{camera_index}_RIGHT", camera_index, x_delta=0.05)
            self.connect_camera_merge_adjustment_button(6, f"CAM{camera_index}_UP", camera_index, y_delta=-0.05)
            self.connect_camera_merge_adjustment_button(6, f"CAM{camera_index}_DOWN", camera_index, y_delta=0.05)
            self.connect_camera_merge_adjustment_button(6, f"CAM{camera_index}_RLEFT", camera_index, rotation_delta=-0.1)
            self.connect_camera_merge_adjustment_button(6, f"CAM{camera_index}_RRIGHT", camera_index, rotation_delta=0.1)

        self.siostatus_server = [self.stackedWidget.widget(i).findChild(QLabel, "status_sio") for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 22, 23]]

        # self.inspection_widget_indices = [8]

        # for i in self.inspection_widget_indices:
        # self.Inspect_button = self.stackedWidget.widget(8).findChild(QPushButton, "InspectButton")
        # if self.Inspect_button:
        #     self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))

        self.inspection_widget_indices = [8, 9]

        for i in self.inspection_widget_indices:
            self.Inspect_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectButton")
            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))


        for i in [8, 9]:
            self.connect_inspectionConfig_button(i, "kansei_plus", "kansei_plus", True)
            self.connect_inspectionConfig_button(i, "kansei_minus", "kansei_minus", True)
            self.connect_inspectionConfig_button(i, "furyou_plus", "furyou_plus", True)
            self.connect_inspectionConfig_button(i, "furyou_minus", "furyou_minus", True)
            self.connect_inspectionConfig_button(i, "kansei_plus_10", "kansei_plus_10", True)
            self.connect_inspectionConfig_button(i, "kansei_minus_10", "kansei_minus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_plus_10", "furyou_plus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_minus_10", "furyou_minus_10", True)
            #connect reset button
            self.connect_inspectionConfig_button(i, "counterReset", "counterReset", True)
            for part_index in range(1, 6):
                self.connect_crop_adjustment_button(i, f"P{part_index}UP", part_index, -1)
                self.connect_crop_adjustment_button(i, f"P{part_index}DOWN", part_index, 1)

        self.connect_line_edit_text_changed(widget_index=8, line_edit_name="kensain_name", inspection_param="kensainNumber")


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
                button_main_menu.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 0))
                dailytenken01_P65820W030P_kanryou_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                # button_dailytenken_kanryou.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 0))

        # self.stackedWidget.currentChanged.connect(self._on_widget_changed)

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()
        self.inspection_thread.emit_crop_values()



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

    def simulateButtonKensaClicks(self):
        self.button_kensa3.click()
        self.button_kensa4.click()

    def connect_inspectionConfig_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_inspection_params(self.inspection_thread, cam_param, value))
            # print(f"Button '{button_name}' connected to cam_param '{cam_param}' with value '{value}' in widget {widget_index}")

    def connect_crop_adjustment_button(self, widget_index, button_name, part_index, delta):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.clicked.connect(
                lambda checked=False, widget_index=widget_index, part_index=part_index, delta=delta:
                self.inspection_thread.adjust_crop_position(widget_index, part_index, delta)
            )

    def connect_camera_merge_adjustment_button(self, widget_index, button_name, camera_index, x_delta=0.0, y_delta=0.0, rotation_delta=0.0):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.clicked.connect(
                lambda checked=False, camera_index=camera_index, x_delta=x_delta, y_delta=y_delta, rotation_delta=rotation_delta:
                self.calibration_thread.adjust_homography_alignment(
                    camera_index,
                    x_delta=x_delta,
                    y_delta=y_delta,
                    rotation_delta=rotation_delta,
                )
            )


    def _close_app(self):
        # self.cam_thread.stop()
        self.calibration_thread.stop()
        self.inspection_thread.stop()
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

    def _setCalibFrame(self, image):
        for i in [1, 2, 3, 4, 5]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame1(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge1")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame2(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge2")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame3(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge3")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame4(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge4")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame5(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge5")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrameAll(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMergeAll")
        label.setPixmap(QPixmap.fromImage(image))

    def _dailyTenkenFrame(self, image):
        widget = self.stackedWidget.widget(21)
        label1 = widget.findChild(QLabel, "dailytenkenFrame")
        label1.setPixmap(QPixmap.fromImage(image))
        
    def _setPartFrame1(self, image):
        for i in [8, 9]:
            widget = self.stackedWidget.widget(i)
            label1 = widget.findChild(QLabel, "FramePart1")
            label1.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame2(self, image):
        for i in [8, 9]:
            widget = self.stackedWidget.widget(i)
            label2 = widget.findChild(QLabel, "FramePart2")
            label2.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame3(self, image):
        for i in [8, 9]:
            widget = self.stackedWidget.widget(i)
            label3 = widget.findChild(QLabel, "FramePart3")
            label3.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame4(self, image):
        for i in [8, 9]:
            widget = self.stackedWidget.widget(i)
            label4 = widget.findChild(QLabel, "FramePart4")
            label4.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame5(self, image):
        for i in [8, 9]:
            widget = self.stackedWidget.widget(i)
            label5 = widget.findChild(QLabel, "FramePart5")
            label5.setPixmap(QPixmap.fromImage(image))

    def _setCropValueLabels(self, widget_index, crop_values):
        widget = self.stackedWidget.widget(widget_index)
        if widget is None:
            return

        for part_index, crop_value in enumerate(crop_values, start=1):
            label = widget.findChild(QLabel, f"CROPVAL_{part_index}")
            if label:
                label.setText(str(crop_value))

    def _setHoleFrame1(self, image):
        widget = self.stackedWidget.widget(8)
        label1 = widget.findChild(QLabel, "MizuAnaPart1")
        label1.setPixmap(QPixmap.fromImage(image))
        
    def _setHoleFrame2(self, image):
        widget = self.stackedWidget.widget(8)
        label2 = widget.findChild(QLabel, "MizuAnaPart2")
        label2.setPixmap(QPixmap.fromImage(image))

    def _setHoleFrame3(self, image):
        widget = self.stackedWidget.widget(8)
        label3 = widget.findChild(QLabel, "MizuAnaPart3")
        label3.setPixmap(QPixmap.fromImage(image))

    def _setHoleFrame4(self, image):
        widget = self.stackedWidget.widget(8)
        label4 = widget.findChild(QLabel, "MizuAnaPart4")
        label4.setPixmap(QPixmap.fromImage(image))

    def _setHoleFrame5(self, image):
        widget = self.stackedWidget.widget(8)
        label5 = widget.findChild(QLabel, "MizuAnaPart5")
        label5.setPixmap(QPixmap.fromImage(image))

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

    def _measurement_label_groups(self, label_count):
        part_suffixes = ("A", "B", "C", "D", "E")
        return [
            [f"R_{part_suffix}_{label_index}" for label_index in range(1, label_count + 1)]
            for part_suffix in part_suffixes
        ]

    def _measurement_text_color(self, pitch_result):
        if pitch_result == 1:
            return "green"
        if pitch_result == 0:
            return "red"
        return "white"

    def _parse_stylesheet_declarations(self, stylesheet):
        declarations = {}
        for rule in stylesheet.split(";"):
            if ":" not in rule:
                continue
            key, value = rule.split(":", 1)
            declarations[key.strip()] = value.strip()
        return declarations

    def _apply_measurement_label_style(self, label, pitch_result):
        base_stylesheet = label.property("baseMeasurementStyle")
        if base_stylesheet is None:
            base_stylesheet = label.styleSheet() or ""
            label.setProperty("baseMeasurementStyle", base_stylesheet)

        declarations = self._parse_stylesheet_declarations(base_stylesheet)
        declarations["background-color"] = self._measurement_text_color(pitch_result)
        declarations["color"] = "black"

        merged_stylesheet = ";\n".join(
            f"{key}: {value}" for key, value in declarations.items()
        )
        if merged_stylesheet:
            merged_stylesheet += ";"

        label.setStyleSheet(merged_stylesheet)

    def _update_measurement_text_labels(self, widget_index, label_count, measurement_value):
        all_label_names = self._measurement_label_groups(label_count)

        for part_index, labels in enumerate(all_label_names):
            if part_index >= len(measurement_value) or measurement_value[part_index] is None:
                part_measurements = [0] * len(labels)
            else:
                part_measurements = list(measurement_value[part_index])

            if len(part_measurements) < len(labels):
                part_measurements.extend([0] * (len(labels) - len(part_measurements)))

            for label_index, label_name in enumerate(labels):
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    label.setText(str(part_measurements[label_index]))

    def _update_measurement_text_colors(self, widget_index, label_count, pitch_result):
        all_label_names = self._measurement_label_groups(label_count)

        for part_index, labels in enumerate(all_label_names):
            if part_index >= len(pitch_result) or pitch_result[part_index] is None:
                part_results = [None] * len(labels)
            else:
                part_results = list(pitch_result[part_index])

            if len(part_results) < len(labels):
                part_results.extend([None] * (len(labels) - len(part_results)))

            for label_index, label_name in enumerate(labels):
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    self._apply_measurement_label_style(label, part_results[label_index])

    def _update_OKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "current_kansei")
                    current_furyou_label = widget.findChild(QLabel, "current_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for numofPart")

    def _update_todayOKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "status_kansei")
                    current_furyou_label = widget.findChild(QLabel, "status_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for todaynumofPart")


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

    def _inspectionStatusText(self, inspectionStatus):
        label_names = ["StatusP1", "StatusP2", "StatusP3", "StatusP4", "StatusP5"]

        for i, status in enumerate(inspectionStatus):
            widget = self.stackedWidget.widget(8)
            label = widget.findChild(QLabel, label_names[i])
            if label:
                label.setText(status)
                if status == "製品検出済み":
                    label.setStyleSheet("QLabel { background-color: lightblue; }")
                elif status == "製品未検出":
                    label.setStyleSheet("QLabel { background-color: pink; }")
                elif status == "OK":
                    label.setStyleSheet("QLabel { background-color: green; }")
                elif status == "NG":
                    label.setStyleSheet("QLabel { background-color: red; }")

    def _P658207YA0A_inspectionStatusText(self, inspectionStatus):
        label_names = ["StatusP1", "StatusP2", "StatusP3", "StatusP4", "StatusP5"]

        for i, status in enumerate(inspectionStatus):
            widget = self.stackedWidget.widget(9)
            label = widget.findChild(QLabel, label_names[i])
            if label:
                label.setText(status)
                if status == "製品検出済み":
                    label.setStyleSheet("QLabel { background-color: lightblue; }")
                elif status == "製品未検出":
                    label.setStyleSheet("QLabel { background-color: pink; }")
                elif status == "検査準備完了":
                    label.setStyleSheet("QLabel { background-color: lightblue; }")
                elif status == "検査中":
                    label.setStyleSheet("QLabel { background-color: pink; }")
                elif status == "OK":
                    label.setStyleSheet("QLabel { background-color: green; }")
                elif status == "NG":
                    label.setStyleSheet("QLabel { background-color: red; }")

    def _inspectionStatusHole(self, holeStatus):
        label_names = ["MizuAnaStatus1", "MizuAnaStatus2", "MizuAnaStatus3", "MizuAnaStatus4", "MizuAnaStatus5"]
        # print(holeStatus)
        for i, status in enumerate(holeStatus):
            widget = self.stackedWidget.widget(8)
            label = widget.findChild(QLabel, label_names[i])
            if label:
                if status == 1:
                    label.setStyleSheet("QLabel { background-color: green; }")
                elif status == 0:
                    label.setStyleSheet("QLabel { background-color: red; }")
                else:
                    label.setStyleSheet("QLabel { background-color: yellow; }")

    def _outputMeasurementText(self, measurementValue):
        self._update_measurement_text_labels(8, 27, measurementValue)

    def _outputMeasurementTextColor(self, pitchResult):
        self._update_measurement_text_colors(8, 27, pitchResult)


    def _P658207YA0A_outputMeasurementText(self, measurementValue):
        self._update_measurement_text_labels(9, 35, measurementValue)

    def _P658207YA0A_outputMeasurementTextColor(self, pitchResult):
        self._update_measurement_text_colors(9, 35, pitchResult)


    def _sentInputToInspectionThread(self, input):
        # set the input[0] till input [4] as inspection thread.inspection_config.kouden_sensor[] and input [5] as inspection thread.inspection_config.button_sensor
        for i in range(5):
            self.inspection_thread.inspection_config.kouden_sensor[i] = input[i]

    def _setEthernetStatusTenmetsuRed(self, input):
        self.server_monitor_thread.server_config.eth_flag_0_4 = input

    def _setEthernetStatusHoldGreen(self, input):
        self.server_monitor_thread.server_config.eth_flag_10_14 = input

    def _setEthernetStatusHoldRed(self, input):
        self.server_monitor_thread.server_config.eth_flag_5_9 = input

    def _set_calib_params(self, thread, key, value):
        setattr(thread.calib_config, key, value)

    def _set_inspection_params(self, thread, key, value):
        setattr(thread.inspection_config, key, value)

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_inspection_params(self.inspection_thread, inspection_param, text))


def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
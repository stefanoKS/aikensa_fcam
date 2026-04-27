import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging
import sqlite3
import mysql.connector

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.camscripts.cam_hole_init import initialize_hole_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound

from ultralytics import YOLO
from aikensa.parts_config.hoodFR_65820W030P import partcheck
from aikensa.parts_config.P658207YA0A_SEALASSYHOODFR import partcheck as P658207YA0A_partcheck

from PIL import ImageFont, ImageDraw, Image

from aikensa.tools.yolo_tools import remove_imageborder_yolo
from aikensa.tools.opencv_tools import add_imageborder

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

    mapCalculated: list = field(default_factory=lambda: [False]*10) #for 10 cameras
    map1: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    map1_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    doInspection: bool = False

    kouden_sensor: list =  field(default_factory=lambda: [0]*5)
    button_sensor: int = 0

    kensainNumber: str = None
    furyou_plus: bool = False
    furyou_minus: bool = False
    kansei_plus: bool = False
    kansei_minus: bool = False
    furyou_plus_10: bool = False #to add 10
    furyou_minus_10: bool = False
    kansei_plus_10: bool = False
    kansei_minus_10: bool = False

    counterReset: bool = False

    today_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])
    current_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])



class InspectionThread(QThread):

    part1Cam = pyqtSignal(QImage)
    part2Cam = pyqtSignal(QImage)
    part3Cam = pyqtSignal(QImage)
    part4Cam = pyqtSignal(QImage)
    part5Cam = pyqtSignal(QImage)

    hole1Cam = pyqtSignal(QImage)
    hole2Cam = pyqtSignal(QImage)
    hole3Cam = pyqtSignal(QImage)
    hole4Cam = pyqtSignal(QImage)
    hole5Cam = pyqtSignal(QImage)

    dailytenkenCam = pyqtSignal(QImage)

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)
    
    hoodFR_InspectionResult_PitchMeasured = pyqtSignal(list)
    P658207YA0A_InspectionResult_PitchMeasured = pyqtSignal(list)

    hoodFR_InspectionResult_PitchResult = pyqtSignal(list)
    P658207YA0A_InspectionResult_PitchResult = pyqtSignal(list)
    
    hoodFR_InspectionStatus = pyqtSignal(list)
    P658207YA0A_InspectionStatus = pyqtSignal(list)

    hoodFR_HoleStatus = pyqtSignal(list)

    ethernet_status_red_tenmetsu = pyqtSignal(list)
    ethernet_status_green_hold = pyqtSignal(list)
    ethernet_status_red_hold = pyqtSignal(list)
    cropPositionChanged = pyqtSignal(int, list)


    

    def __init__(self, inspection_config: InspectionConfig = None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        #Initialize AI checkpoint


        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.multiCam_stream = False

        self.cap_cam = None
        self.cap_cam0 = None
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None
        self.cap_cam4 = None
        self.cap_cam5 = None

        self.bottomframe = None
        self.mergeframe1 = None
        self.mergeframe2 = None
        self.mergeframe3 = None
        self.mergeframe4 = None
        self.mergeframe5 = None
        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None
        self.mergeframe3_scaled = None
        self.mergeframe4_scaled = None
        self.mergeframe5_scaled = None
        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None
        self.mergeframe3_downsampled = None
        self.mergeframe4_downsampled = None
        self.mergeframe5_downsampled = None

        self.holeFrame1 = None
        self.holeFrame2 = None
        self.holeFrame3 = None
        self.holeFrame4 = None
        self.holeFrame5 = None

        self.holeImageMerge = [None]*5

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        self.homography_matrix3 = None
        self.homography_matrix4 = None
        self.homography_matrix5 = None
        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None
        self.homography_matrix3_scaled = None
        self.homography_matrix4_scaled = None
        self.homography_matrix5_scaled = None
        self.H1 = None
        self.H2 = None
        self.H3 = None
        self.H4 = None
        self.H5 = None
        self.H1_scaled = None
        self.H2_scaled = None
        self.H3_scaled = None
        self.H4_scaled = None
        self.H5_scaled = None
        self.H1_FR = None
        self.H2_FR = None
        self.H3_FR = None
        self.H4_FR = None
        self.H5_FR = None
        self.H1_scaled_FR = None
        self.H2_scaled_FR = None
        self.H3_scaled_FR = None
        self.H4_scaled_FR = None
        self.H5_scaled_FR = None
        self.homography_adjustment_FR = {}

        self.part1Crop = None
        self.part2Crop = None
        self.part3Crop = None
        self.part4Crop = None
        self.part5Crop = None
        
        self.part1Crop_scaled = None
        self.part2Crop_scaled = None
        self.part3Crop_scaled = None
        self.part4Crop_scaled = None
        self.part5Crop_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.scale_factor = 5.0 #Scale Factor, might increase this later
        self.scale_factor_hole = 2.0
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.planarizeTransform = None
        self.planarizeTransform_scaled = None

        self.planarizeTransform_temp = None
        self.planarizeTransform_temp_scaled = None
        self.homographyWarpCache = {}
        self.planarizeWarpCache = {}
        self.homography_adjustment_path = os.path.join("aikensa", "cameracalibration", "homography_adjustment.yaml")
        self.crop_settings_path = os.path.join("aikensa", "inspection", "crop_settings.yaml")
        self.crop_position_profiles = {
            "65820W030P": {
                "attrs": [
                    "part1Crop_YPos",
                    "part2Crop_YPos",
                    "part3Crop_YPos",
                    "part4Crop_YPos",
                    "part5Crop_YPos",
                ],
                "defaults": [45, 300, 580, 860, 1130],
            },
            "658207YA0A": {
                "attrs": [
                    "part1Crop_YPos_hoodFR",
                    "part2Crop_YPos_hoodFR",
                    "part3Crop_YPos_hoodFR",
                    "part4Crop_YPos_hoodFR",
                    "part5Crop_YPos_hoodFR",
                ],
                "defaults": [63 * 5, 101 * 5, 139 * 5, 174 * 5, 212 * 5],
            },
        }
        
        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        self.part_height_offset = 110
        self.part_height_offset_scaled = int(self.part_height_offset//self.scale_factor)

        self.part_height_offset_hoodFR = 140
        self.part_height_offset_hoodFR_scaled = int(self.part_height_offset_hoodFR//self.scale_factor)

        self.part_height_offset_nissanhoodFR = 180
        self.part_height_offset_nissanhoodFR_scaled = int(self.part_height_offset_nissanhoodFR//self.scale_factor)

        self.dailyTenken_cropWidth = 950
        self.dailyTenken_cropWidth_scaled = int(self.dailyTenken_cropWidth//self.scale_factor)

        for profile_config in self.crop_position_profiles.values():
            for attr_name, default_value in zip(profile_config["attrs"], profile_config["defaults"]):
                setattr(self, attr_name, default_value)

        self.refresh_crop_position_cache()
        self.load_crop_settings()
        
        self.hole1Crop_XYpos_scaled = (int(70//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole2Crop_XYpos_scaled = (int(295//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole3Crop_XYpos_scaled = (int(525//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole4Crop_XYpos_scaled = (int(760//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole5Crop_XYpos_scaled = (int(980//self.scale_factor_hole), int(180//self.scale_factor_hole))

        self.height_hole_offset = int(120//self.scale_factor_hole)
        self.width_hole_offset = int(370//self.scale_factor_hole)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*5

        self.InspectionImages_endSegmentation_Left = [None]*5
        self.InspectionImages_endSegmentation_Right = [None]*5
        
        self.InspectionResult_ClipDetection = [None]*5

        self.InspectionResult_EndSegmentation_Left = [None]*5
        self.InspectionResult_EndSegmentation_Right = [None]*5

        self.InspectionResult_HoleDetection = [None]*5

        self.InspectionResult_PitchMeasured = [None]*5
        self.InspectionResult_PitchResult = [None]*5
        self.InspectionResult_DetectionID = [None]*5
        self.InspectionResult_Status = [None]*5
        self.InspectionResult_DeltaPitch = [None]*30
        self.InspectionResult_NGReason = [None]*5
        
        self.InspectionStatus = [None]*5

        self.DetectionResult_HoleDetection = [None]*5

        self.ethernet_status_red_tenmetsu_status = [0]*5
        self.ethernet_status_green_hold_status = [0]*5
        self.ethernet_status_red_hold_status = [0]*5

        self.ethernet_status_red_tenmetsu_status_prev = [0]*5
        self.ethernet_status_green_hold_status_prev = [0]*5
        self.ethernet_status_red_hold_status_prev = [0]*5

        self.InspectionImages_prev = [None]*5
        self.hoodFR_InspectionResult_PitchMeasured_prev = None
        self.hoodFR_InspectionResult_PitchResult_prev =None
        self.hoodFR_InspectionStatus_prev = None

        self._test = [0]*5

        self.hole_detection = [None] * 5
        self.hole_detection_result = [None] * 5

        self.widget_dir_map = {
            8: "65820W030P",
            9: "658207YA0A",
        }

        self.InspectionWaitTime = 15.0
        self.InspectionTimeStart = None

        self.test = 0
        self.firstTimeInspection = True

                # "Read mysql id and password from yaml file"
        with open("aikensa/mysql/id.yaml") as file:
            credentials = yaml.load(file, Loader=yaml.FullLoader)
            self.mysqlID = credentials["id"]
            self.mysqlPassword = credentials["pass"]
            self.mysqlHost = credentials["host"]
            self.mysqlHostPort = credentials["port"]


    def release_all_camera(self):
        if self.cap_cam0 is not None:
            self.cap_cam0.release()
            print(f"Camera 0 released.")
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")
        if self.cap_cam5 is not None:
            self.cap_cam5.release()
            print(f"Camera 5 released.")

    def initialize_single_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.inspection_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            # self.frame = self.create_placeholder_image()
        else:
            self.cap_cam = initialize_camera(camID)
            if not self.cap_cam.isOpened():
                print(f"Failed to open camera with ID {camID}")
                self.cap_cam = None
            else:
                print(f"Initialized Camera on ID {camID}")

    def initialize_all_camera(self):
        if self.cap_cam0 is not None:
            self.cap_cam0.release()
            print(f"Camera 0 released.")
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")
        if self.cap_cam5 is not None:
            self.cap_cam5.release()
            print(f"Camera 5 released.")
        
        self.cap_cam0 = initialize_hole_camera(0)
        self.cap_cam1 = initialize_camera(1)
        self.cap_cam2 = initialize_camera(2)
        self.cap_cam3 = initialize_camera(3)
        self.cap_cam4 = initialize_camera(4)
        self.cap_cam5 = initialize_camera(5)

        if not self.cap_cam0.isOpened():
            print(f"Failed to open camera with ID 0")
            self.cap_cam0 = None
        else:
            print(f"Initialized Camera on ID 0")
        if not self.cap_cam1.isOpened():
            print(f"Failed to open camera with ID 1")
            self.cap_cam1 = None
        else:
            print(f"Initialized Camera on ID 1")
        if not self.cap_cam2.isOpened():
            print(f"Failed to open camera with ID 2")
            self.cap_cam2 = None
        else:
            print(f"Initialized Camera on ID 2")
        if not self.cap_cam3.isOpened():
            print(f"Failed to open camera with ID 3")
            self.cap_cam3 = None
        else:
            print(f"Initialized Camera on ID 3")
        if not self.cap_cam4.isOpened():
            print(f"Failed to open camera with ID 4")
            self.cap_cam4 = None
        else:
            print(f"Initialized Camera on ID 4")
        if not self.cap_cam5.isOpened():
            print(f"Failed to open camera with ID 5")
            self.cap_cam5 = None
        else:
            print(f"Initialized Camera on ID 5")

    def run(self):
        #initialize the database
        if not os.path.exists("./aikensa/inspection_results"):
            os.makedirs("./aikensa/inspection_results")

        self.conn = sqlite3.connect('./aikensa/inspection_results/database_results.db')
        self.cursor = self.conn.cursor()
        # Create the table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partName TEXT,
            numofPart TEXT,
            currentnumofPart TEXT,
            timestampHour TEXT,
            timestampDate TEXT,
            deltaTime REAL,
            kensainName TEXT,
            detected_pitch TEXT,
            delta_pitch TEXT,
            total_length REAL
        )
        ''')

        # List of columns to add
        columns_to_add = [
            ("resultpitch", "TEXT"),
            ("status", "TEXT"),
            ("NGreason", "TEXT")
        ]

        # Using the function to add columns
        self.add_columns(self.cursor, "inspection_results", columns_to_add)

        self.conn.commit()


        #Initialize connection to mysql server if available
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.mysqlHost,
                user=self.mysqlID,
                password=self.mysqlPassword,
                port=self.mysqlHostPort,
                database="AIKENSAresults"
            )
            print(f"Connected to MySQL database at {self.mysqlHost}")
        except Exception as e:
            print(f"Error connecting to MySQL database: {e}")
            self.mysql_conn = None

        #try adding data to the schema in mysql
        if self.mysql_conn is not None:
            self.mysql_cursor = self.mysql_conn.cursor()
            self.mysql_cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_results (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                partName TEXT,
                numofPart TEXT,
                currentnumofPart TEXT,
                timestampHour TEXT,
                timestampDate TEXT,
                deltaTime REAL,
                kensainName TEXT,
                detected_pitch TEXT,
                delta_pitch TEXT,
                total_length REAL,
                resultpitch TEXT,
                status TEXT,
                NGreason TEXT
            )
            ''')
            self.mysql_conn.commit()




        #print thread started
        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Model Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"
        self.homography_adjustment_fr_path = os.path.join(self._save_dir, "homography_adjustment_FR.yaml")

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        for key, value in self.widget_dir_map.items():
            self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)


        #INIT all variables

        self.load_homography_set()
        self.load_homography_set(suffix="_FR")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform = np.array(transform_list)
                print(f"Loaded planarizeTransform matrix")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_scaled = np.array(transform_list)
                print(f"Loaded scaled planarizeTransform matrix")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_temp.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_temp.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_temp = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_temp_scaled = np.array(transform_list)     

        self.cache_perspective_maps()


        while self.running:

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget == 8:
                if self._handle_widget_8():
                    continue

            if self.inspection_config.widget == 9:
                if self._handle_widget_9():
                    continue



            if self.inspection_config.widget == 21:
                if self._handle_widget_21():
                    continue

        self.msleep(1)

    def _handle_widget_21(self):

        if self.multiCam_stream is False:
            self.multiCam_stream = True
            self.initialize_all_camera()

        if not self.has_required_cameras([3]):
            self.emit_blank_dailytenken_frame()
            self.msleep(100)
            return True

        self.mergeframe3 = self.read_camera_frame(self.cap_cam3)
        if self.mergeframe3 is None:
            self.emit_blank_dailytenken_frame()
            self.msleep(100)
            return True

        self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)

        #Downsampled the image
        self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)

        if self.inspection_config.mapCalculated[1] is False: #Just checking the first camera to reduce loop time
            for i in range(1, 6):
                if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                    camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                    # Precompute the undistort and rectify map for faster processing
                    h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                    self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                    print(f"map1 and map2 value is calculated")
                    self.inspection_config.mapCalculated[i] = True
                    print(f"Calibration map is calculated for Camera {i}")

                    #Also do the map for the scaled image
                    camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                    h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1 as reference
                    self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                    print(f"map1 and map2 value is calculated for scaled image")
                    print(f"Calibration map is calculated for Camera {i} for scaled image")

                    #Not idea but the condition use the bigger image

        if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

            if self.inspection_config.doInspection is False:

                self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                self.combinedImage_scaled = self.homography_blank_canvas_scaled.copy()
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe3_scaled, "H3_scaled")

                self.combinedImage_scaled = self.apply_planarize_with_cache(self.combinedImage_scaled, "planarizeTransform_scaled", (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                #Crop the image scaled for each part
                self.part3Crop_scaled = self.combinedImage_scaled[self.part3Crop_YPos_scaled : self.part3Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                #crop the image again, taking the center area with the width of self.dailyTenken_cropWidth only
                cv2.imwrite("part3Crop_scaled.png", self.part3Crop_scaled)

                self.part3Crop_scaled = self.part3Crop_scaled[:, int((self.part3Crop_scaled.shape[1] - self.dailyTenken_cropWidth_scaled)/2) : int((self.part3Crop_scaled.shape[1] + self.dailyTenken_cropWidth_scaled)/2)]
                cv2.imwrite("part3Crop2_scaled.png", self.part3Crop_scaled)


                self.part3Crop_scaled = self.downSampling(self.part3Crop_scaled, width=1710, height=198)

                self.InspectionResult_PitchMeasured = [None]*5
                self.InspectionResult_PitchResult = [None]*5

                if self.part3Crop_scaled is not None:
                    self.dailytenkenCam.emit(self.convertQImage(self.part3Crop_scaled))

            if self.InspectionTimeStart is None:
                self.InspectionTimeStart = time.time()

            # print(time.time() - self.InspectionTimeStart)

            if self.firstTimeInspection is False:
                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

            # print(f"InspectionConfig: {self.inspection_config.doInspection}")

            if self.inspection_config.doInspection is True:
                if not self.has_required_models(21):
                    self.inspection_config.doInspection = False
                    self.emit_blank_dailytenken_frame()
                    print("Skipping widget 21 inspection because required model weights are missing.")
                    return True

                self.inspection_config.doInspection = False

                # if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                #     #Break the bottom if
                #     print("No Kensain Number Input")
                #     for i in range (len(self.InspectionStatus)):
                #         self.InspectionStatus[i] = "社員番号未入力"
                #     self.hoodFR_InspectionStatus.emit(self.InspectionStatus)
                #     continue

                if self.InspectionTimeStart is not None:

                    if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:
                        self.firstTimeInspection is False
                        print("Inspection Started")
                        self.InspectionTimeStart = time.time()

                        self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage = self.homography_blank_canvas.copy()
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe3, "H3")

                        self.combinedImage = self.apply_planarize_with_cache(self.combinedImage, "planarizeTransform", (int(self.homography_size[1]), int(self.homography_size[0])))
                        self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/(1.26))))#1.48 for the qt, 1.26 for the aspect ratio

                        # Crop the image scaled for each part

                        self.part3Crop = self.combinedImage_scaled[self.part3Crop_YPos : self.part3Crop_YPos + self.part_height_offset, 0 : self.homography_size[1]]


                        #Need to convert to BGR for SAHI Inspection
                        self.part1Crop = cv2.cvtColor(self.part1Crop, cv2.COLOR_RGB2BGR)
                        self.part2Crop = cv2.cvtColor(self.part2Crop, cv2.COLOR_RGB2BGR)
                        self.part3Crop = cv2.cvtColor(self.part3Crop, cv2.COLOR_RGB2BGR)
                        self.part4Crop = cv2.cvtColor(self.part4Crop, cv2.COLOR_RGB2BGR)
                        self.part5Crop = cv2.cvtColor(self.part5Crop, cv2.COLOR_RGB2BGR)

                        #Put the All the image into a list
                        self.InspectionImages[0] = self.part1Crop.copy()
                        self.InspectionImages[1] = self.part2Crop.copy()
                        self.InspectionImages[2] = self.part3Crop.copy()
                        self.InspectionImages[3] = self.part4Crop.copy()
                        self.InspectionImages[4] = self.part5Crop.copy()

                        print(f"Length of Inspection Images : {len(self.InspectionImages)}")

                        #Emit　検査中 to the status signal

                        # # Do the inspection
                        for i in range(len(self.InspectionImages)):
                            #Only do inspectino on the one with kouden sensor on
                            if self.inspection_config.kouden_sensor[i] == 1:
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages[i],
                                    self.hoodFR_clipDetectionModel,
                                    slice_height=180, slice_width=1280,
                                    overlap_height_ratio=0.0, overlap_width_ratio=0.3,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True
                                )

                                #Crop image from 0 to 1024 width for the left and width-1024 to 1024 for the right
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1024, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1024:, :]

                                self.InspectionResult_EndSegmentation_Left[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False)
                                self.InspectionResult_EndSegmentation_Right[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = partcheck(self.InspectionImages[i],
                                                                                                                                                                          self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                          self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                          self.InspectionResult_EndSegmentation_Right[i])
                            else:
                                #Make pure black image
                                self.InspectionImages[i] = np.full((24, 1771, 3), (10, 10, 20), dtype=np.uint8)
                                self.InspectionResult_PitchMeasured[i] = None
                                self.InspectionResult_PitchResult[i] = None
                                self.InspectionResult_DetectionID[i] = None
                                self.InspectionResult_Status[i] = None

                            # print(self.InspectionResult_Status[i])
                            # print(self.InspectionResult_DetectionID[i])

                            if self.InspectionResult_Status[i] == "OK":
                                self.ethernet_status_green_hold_status[i] = 1
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 0

                                self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1

                                #Play konpou sound if the current_numofPart is dividable by 25
                                if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 25 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                                    play_konpou_sound()

                                self.InspectionStatus[i] = "OK"

                            elif self.InspectionResult_Status[i] == "NG":
                                self.ethernet_status_green_hold_status[i] = 0
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 1
                                self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1

                                self.InspectionStatus[i] = "NG"

                            else:
                                self.ethernet_status_green_hold_status[i] = 0
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 0

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                                currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                deltaTime = 0.0,
                                kensainName = self.inspection_config.kensainNumber,
                                detected_pitch_str = self.InspectionResult_PitchMeasured[i],
                                delta_pitch_str = self.InspectionResult_DeltaPitch[i],
                                total_length=0,
                                resultPitch = self.InspectionResult_PitchResult[i],
                                status = self.InspectionStatus[i],
                                NGreason = self.InspectionResult_NGReason[i])

                            #save hole image
                            self.save_image_hole(self.holeFrame1, False, "P1")
                            self.save_image_hole(self.holeFrame2, False, "P2")
                            self.save_image_hole(self.holeFrame3, False, "P3")
                            self.save_image_hole(self.holeFrame4, False, "P4")
                            self.save_image_hole(self.holeFrame5, False, "P5")

                            self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                        self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                        self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                        self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                        self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                        self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                        self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1771, height=24)
                        self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1771, height=24)
                        self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1771, height=24)
                        self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1771, height=24)
                        self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1771, height=24)

                        self.hoodFR_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                        self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)
                        print("Inspection Finished")
                        #Remember that list is mutable
                        self.ethernet_status_red_tenmetsu_status_prev = self.ethernet_status_red_tenmetsu_status_prev.copy()
                        self.ethernet_status_green_hold_status_prev = self.ethernet_status_green_hold_status.copy()
                        self.ethernet_status_red_hold_status_prev = self.ethernet_status_red_hold_status.copy()

                        self.InspectionImages_prev[0] = self.InspectionImages[0]
                        self.InspectionImages_prev[1] = self.InspectionImages[1]
                        self.InspectionImages_prev[2] = self.InspectionImages[2]
                        self.InspectionImages_prev[3] = self.InspectionImages[3]
                        self.InspectionImages_prev[4] = self.InspectionImages[4]

                        self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                        self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()
                        self.InspectionStatus_prev = self.InspectionStatus.copy()

                        self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                        self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                        self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                        self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                        self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                        self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                        self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                        self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                        # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

        #emit the ethernet
        self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
        self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
        self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
        self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

        # Emit status based on the red tenmetsu status

        self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

        # Emit the hole detection
        self.hoodFR_HoleStatus.emit(self.DetectionResult_HoleDetection)
        return False

    def _handle_widget_9(self):

        if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
            self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                self.inspection_config.furyou_plus,
                self.inspection_config.furyou_minus,
                self.inspection_config.furyou_plus_10,
                self.inspection_config.furyou_minus_10,
                self.inspection_config.kansei_plus,
                self.inspection_config.kansei_minus,
                self.inspection_config.kansei_plus_10,
                self.inspection_config.kansei_minus_10)
            print("Manual Adjustment Done")
            print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
            print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
            print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
            print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
            print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
            print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
            print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
            print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")

        if self.inspection_config.counterReset is True:
            self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
            self.inspection_config.counterReset = False
            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                    currentnumofPart = [0, 0],
                    deltaTime = 0.0,
                    kensainName = self.inspection_config.kensainNumber,
                    detected_pitch_str = "COUNTERRESET",
                    delta_pitch_str = "COUNTERRESET",
                    total_length=0,
                    resultPitch = "COUNTERRESET",
                    status = "COUNTERRESET",
                    NGreason = "COUNTERRESET")
        #check kouden sensor and tenmetsu status

        for i in range (len(self.inspection_config.kouden_sensor)):
            self.ethernet_status_red_tenmetsu_status[i] = self.inspection_config.kouden_sensor[i]
            self.ethernet_status_green_hold_status[i] = 0 #Reset at every loop
            self.ethernet_status_red_hold_status[i] = 0 #Reset at every loop

            self.InspectionStatus[i] = "検査準備完了"
            self.InspectionStatus[i] = "検査準備完了"

        if self.multiCam_stream is False:
            self.multiCam_stream = True
            self.initialize_all_camera()

        if not self.has_required_cameras([1, 2, 3, 4, 5]):
            self.emit_blank_part_frames()
            self.msleep(100)
            return True

        # _, self.bottomframe = self.cap_cam0.read()
        self.mergeframe1 = self.read_camera_frame(self.cap_cam1)
        self.mergeframe2 = self.read_camera_frame(self.cap_cam2)
        self.mergeframe3 = self.read_camera_frame(self.cap_cam3)
        self.mergeframe4 = self.read_camera_frame(self.cap_cam4)
        self.mergeframe5 = self.read_camera_frame(self.cap_cam5)

        if any(frame is None for frame in (self.mergeframe1, self.mergeframe2, self.mergeframe3, self.mergeframe4, self.mergeframe5)):
            self.emit_blank_part_frames()
            self.msleep(100)
            return True

        self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
        self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
        self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
        self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
        self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)

        #Downsampled the image
        self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
        self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)
        self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)
        self.mergeframe4_scaled = self.downSampling(self.mergeframe4, self.scaled_width, self.scaled_height)
        self.mergeframe5_scaled = self.downSampling(self.mergeframe5, self.scaled_width, self.scaled_height)

        if self.inspection_config.mapCalculated[1] is False: #Just checking the first camera to reduce loop time
            for i in range(1, 6):
                if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                    camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                    # Precompute the undistort and rectify map for faster processing
                    h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                    self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                    print(f"map1 and map2 value is calculated")
                    self.inspection_config.mapCalculated[i] = True
                    print(f"Calibration map is calculated for Camera {i}")

                    #Also do the map for the scaled image
                    camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                    h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1 as reference
                    self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                    print(f"map1 and map2 value is calculated for scaled image")
                    print(f"Calibration map is calculated for Camera {i} for scaled image")

                    #Not idea but the condition use the bigger image

        if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

            # #rotate the bottom frame 90deg CCW
            # self.bottomframe = self.downScaledImage(self.bottomframe, self.scale_factor_hole)
            # self.bottomframe = cv2.flip(cv2.rotate(self.bottomframe, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

            # self.holeFrame1 = self.bottomframe[self.hole1Crop_XYpos_scaled[0]:self.hole1Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole1Crop_XYpos_scaled[1]:self.hole1Crop_XYpos_scaled[1] + self.width_hole_offset]
            # self.holeFrame2 = self.bottomframe[self.hole2Crop_XYpos_scaled[0]:self.hole2Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole2Crop_XYpos_scaled[1]:self.hole2Crop_XYpos_scaled[1] + self.width_hole_offset]
            # self.holeFrame3 = self.bottomframe[self.hole3Crop_XYpos_scaled[0]:self.hole3Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole3Crop_XYpos_scaled[1]:self.hole3Crop_XYpos_scaled[1] + self.width_hole_offset]
            # self.holeFrame4 = self.bottomframe[self.hole4Crop_XYpos_scaled[0]:self.hole4Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole4Crop_XYpos_scaled[1]:self.hole4Crop_XYpos_scaled[1] + self.width_hole_offset]
            # self.holeFrame5 = self.bottomframe[self.hole5Crop_XYpos_scaled[0]:self.hole5Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole5Crop_XYpos_scaled[1]:self.hole5Crop_XYpos_scaled[1] + self.width_hole_offset]

            if self.inspection_config.doInspection is False:

                self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.inspection_config.map1_downscaled[4], self.inspection_config.map2_downscaled[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe5_scaled = cv2.remap(self.mergeframe5_scaled, self.inspection_config.map1_downscaled[5], self.inspection_config.map2_downscaled[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                self.combinedImage_scaled = self.homography_blank_canvas_scaled.copy()
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe1_scaled, "H1_scaled_FR")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe2_scaled, "H2_scaled_FR")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe3_scaled, "H3_scaled_FR")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe4_scaled, "H4_scaled_FR")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe5_scaled, "H5_scaled_FR")

                self.combinedImage_scaled = self.apply_planarize_with_cache(self.combinedImage_scaled, "planarizeTransform_temp_scaled", (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                #Crop the image scaled for each part
                self.part1Crop_scaled = self.combinedImage_scaled[self.part1Crop_Ypos_hoodFR_scaled : self.part1Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                self.part2Crop_scaled = self.combinedImage_scaled[self.part2Crop_Ypos_hoodFR_scaled : self.part2Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                self.part3Crop_scaled = self.combinedImage_scaled[self.part3Crop_Ypos_hoodFR_scaled : self.part3Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                self.part4Crop_scaled = self.combinedImage_scaled[self.part4Crop_Ypos_hoodFR_scaled : self.part4Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                self.part5Crop_scaled = self.combinedImage_scaled[self.part5Crop_Ypos_hoodFR_scaled : self.part5Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]

                # cv2.imwrite("part1Crop_scaled_inference.png", self.part1Crop_scaled)

                self.part1Crop_scaled = self.downSampling(self.part1Crop_scaled, width=1771, height=24)
                self.part2Crop_scaled = self.downSampling(self.part2Crop_scaled, width=1771, height=24)
                self.part3Crop_scaled = self.downSampling(self.part3Crop_scaled, width=1771, height=24)
                self.part4Crop_scaled = self.downSampling(self.part4Crop_scaled, width=1771, height=24)
                self.part5Crop_scaled = self.downSampling(self.part5Crop_scaled, width=1771, height=24)

                self.InspectionResult_PitchMeasured = [None]*5
                self.InspectionResult_PitchResult = [None]*5

                if self.part1Crop_scaled is not None:
                    self.part1Cam.emit(self.convertQImage(self.part1Crop_scaled))
                if self.part2Crop_scaled is not None:
                    self.part2Cam.emit(self.convertQImage(self.part2Crop_scaled))
                if self.part3Crop_scaled is not None:
                    self.part3Cam.emit(self.convertQImage(self.part3Crop_scaled))
                if self.part4Crop_scaled is not None:
                    self.part4Cam.emit(self.convertQImage(self.part4Crop_scaled))
                if self.part5Crop_scaled is not None:
                    self.part5Cam.emit(self.convertQImage(self.part5Crop_scaled))

                #Empty the Inspection Result

                self.P658207YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                # self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)

            if self.InspectionTimeStart is None:
                self.InspectionTimeStart = time.time()

            # print(time.time() - self.InspectionTimeStart)

            if self.firstTimeInspection is False:
                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

            # print(f"InspectionConfig: {self.inspection_config.doInspection}")

            if self.inspection_config.doInspection is True:
                if not self.has_required_models(9):
                    self.inspection_config.doInspection = False
                    self.emit_blank_part_frames()
                    print("Skipping widget 9 inspection because required model weights are missing.")
                    return True

                self.inspection_config.doInspection = False

                # if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                #     #Break the bottom if
                #     print("No Kensain Number Input")
                #     for i in range (len(self.InspectionStatus)):
                #         self.InspectionStatus[i] = "社員番号未入力"
                #     self.hoodFR_InspectionStatus.emit(self.InspectionStatus)
                #     continue

                if self.InspectionTimeStart is not None:

                    if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:
                        self.firstTimeInspection is False
                        print("Inspection Started")
                        self.InspectionTimeStart = time.time()

                        for i in range (len(self.InspectionStatus)):
                            self.InspectionStatus[i] = "検査中"
                        self.P658207YA0A_InspectionStatus.emit(self.InspectionStatus)

                        self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[2], self.inspection_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe4 = cv2.remap(self.mergeframe4, self.inspection_config.map1[4], self.inspection_config.map2[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe5 = cv2.remap(self.mergeframe5, self.inspection_config.map1[5], self.inspection_config.map2[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage = self.homography_blank_canvas.copy()
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe1, "H1_FR")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe2, "H2_FR")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe3, "H3_FR")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe4, "H4_FR")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe5, "H5_FR")

                        self.combinedImage = self.apply_planarize_with_cache(self.combinedImage, "planarizeTransform_temp", (int(self.homography_size[1]), int(self.homography_size[0])))
                        self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/(1.26))))#1.48 for the qt, 1.26 for the aspect ratio

                        # Crop the image scaled for each part
                        self.part1Crop = self.combinedImage[int(self.part1Crop_YPos_hoodFR*1.48) : int((self.part1Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part2Crop = self.combinedImage[int(self.part2Crop_YPos_hoodFR*1.48) : int((self.part2Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part3Crop = self.combinedImage[int(self.part3Crop_YPos_hoodFR*1.48) : int((self.part3Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part4Crop = self.combinedImage[int(self.part4Crop_YPos_hoodFR*1.48) : int((self.part4Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part5Crop = self.combinedImage[int(self.part5Crop_YPos_hoodFR*1.48) : int((self.part5Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]

                        # self.save_image(self.part1Crop)
                        # time.sleep(1.5)
                        # self.save_image(self.part2Crop)
                        # time.sleep(1.5)
                        # self.save_image(self.part3Crop)
                        # time.sleep(1.5)
                        # self.save_image(self.part4Crop)
                        # time.sleep(1.5)
                        # self.save_image(self.part5Crop)
                        # time.sleep(1.5)

                        #Need to convert to BGR for SAHI Inspection
                        self.part1Crop = cv2.cvtColor(self.part1Crop, cv2.COLOR_RGB2BGR)
                        self.part2Crop = cv2.cvtColor(self.part2Crop, cv2.COLOR_RGB2BGR)
                        self.part3Crop = cv2.cvtColor(self.part3Crop, cv2.COLOR_RGB2BGR)
                        self.part4Crop = cv2.cvtColor(self.part4Crop, cv2.COLOR_RGB2BGR)
                        self.part5Crop = cv2.cvtColor(self.part5Crop, cv2.COLOR_RGB2BGR)

                        #Put the All the image into a list
                        self.InspectionImages[0] = self.part1Crop.copy()
                        self.InspectionImages[1] = self.part2Crop.copy()
                        self.InspectionImages[2] = self.part3Crop.copy()
                        self.InspectionImages[3] = self.part4Crop.copy()
                        self.InspectionImages[4] = self.part5Crop.copy()

                        print(f"Length of Inspection Images : {len(self.InspectionImages)}")


                        #Emit　検査中 to the status signal

                        # # Do the inspection
                        for i in range(len(self.InspectionImages)):
                            #Only do inspectino on the one with kouden sensor ons
                            self.inspection_config.kouden_sensor[i] = 1 #forcefully make it 1
                            if self.inspection_config.kouden_sensor[i] == 1:
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages[i],
                                    self.P658207YA0A_clipDetectionModel,
                                    slice_height=497, slice_width=1960,
                                    overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=False
                                )

                                #Crop image from 0 to 1024 width for the left and width-1024 to 1024 for the right
                                # self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1024, :]
                                # self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1024:, :]

                                # self.InspectionResult_EndSegmentation_Left[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False)
                                # self.InspectionResult_EndSegmentation_Right[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P658207YA0A_partcheck(self.InspectionImages[i], self.InspectionResult_ClipDetection[i].object_prediction_list)
                            else:
                                #Make pure black image
                                self.InspectionImages[i] = np.full((24, 1771, 3), (10, 10, 20), dtype=np.uint8)
                                self.InspectionResult_PitchMeasured[i] = None
                                self.InspectionResult_PitchResult[i] = None
                                self.InspectionResult_DetectionID[i] = None
                                self.InspectionResult_Status[i] = None

                            # print(self.InspectionResult_Status[i])
                            # print(self.InspectionResult_DetectionID[i])

                            if self.InspectionResult_Status[i] == "OK":
                                self.ethernet_status_green_hold_status[i] = 1
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 0

                                self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1

                                #Play konpou sound if the current_numofPart is dividable by 25
                                if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 25 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                                    play_konpou_sound()

                                self.InspectionStatus[i] = "OK"

                            elif self.InspectionResult_Status[i] == "NG":
                                self.ethernet_status_green_hold_status[i] = 0
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 1
                                self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1

                                self.InspectionStatus[i] = "NG"

                            else:
                                self.ethernet_status_green_hold_status[i] = 0
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 0

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                                currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                deltaTime = 0.0,
                                kensainName = self.inspection_config.kensainNumber,
                                detected_pitch_str = self.InspectionResult_PitchMeasured[i],
                                delta_pitch_str = self.InspectionResult_DeltaPitch[i],
                                total_length=0,
                                resultPitch = self.InspectionResult_PitchResult[i],
                                status = self.InspectionStatus[i],
                                NGreason = self.InspectionResult_NGReason[i])


                            # #save hole image
                            # self.save_image_hole(self.holeFrame1, False, "P1")
                            # self.save_image_hole(self.holeFrame2, False, "P2")
                            # self.save_image_hole(self.holeFrame3, False, "P3")
                            # self.save_image_hole(self.holeFrame4, False, "P4")
                            # self.save_image_hole(self.holeFrame5, False, "P5")

                            self.P658207YA0A_InspectionStatus.emit(self.InspectionStatus)

                        self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                        self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                        self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                        self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                        self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                        self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1771, height=24)
                        self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1771, height=24)
                        self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1771, height=24)
                        self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1771, height=24)
                        self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1771, height=24)

                        self.P658207YA0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                        self.P658207YA0A_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)

                        print("Inspection Finished")
                        #Remember that list is mutable
                        self.ethernet_status_red_tenmetsu_status_prev = self.ethernet_status_red_tenmetsu_status_prev.copy()
                        self.ethernet_status_green_hold_status_prev = self.ethernet_status_green_hold_status.copy()
                        self.ethernet_status_red_hold_status_prev = self.ethernet_status_red_hold_status.copy()

                        self.InspectionImages_prev[0] = self.InspectionImages[0]
                        self.InspectionImages_prev[1] = self.InspectionImages[1]
                        self.InspectionImages_prev[2] = self.InspectionImages[2]
                        self.InspectionImages_prev[3] = self.InspectionImages[3]
                        self.InspectionImages_prev[4] = self.InspectionImages[4]

                        self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                        self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()
                        self.InspectionStatus_prev = self.InspectionStatus.copy()

                        self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                        self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                        self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                        self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                        self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                        self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                        self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                        self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                        time.sleep(10)

                        # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

        #emit the ethernet
        self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
        self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
        self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
        self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

        # Emit status based on the red tenmetsu status

        self.P658207YA0A_InspectionStatus.emit(self.InspectionStatus)
        return False

    def _handle_widget_8(self):

        if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
            self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                self.inspection_config.furyou_plus,
                self.inspection_config.furyou_minus,
                self.inspection_config.furyou_plus_10,
                self.inspection_config.furyou_minus_10,
                self.inspection_config.kansei_plus,
                self.inspection_config.kansei_minus,
                self.inspection_config.kansei_plus_10,
                self.inspection_config.kansei_minus_10)
            print("Manual Adjustment Done")
            print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
            print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
            print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
            print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
            print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
            print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
            print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
            print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")

        if self.inspection_config.counterReset is True:
            self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
            self.inspection_config.counterReset = False
            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                    currentnumofPart = [0, 0],
                    deltaTime = 0.0,
                    kensainName = self.inspection_config.kensainNumber,
                    detected_pitch_str = "COUNTERRESET",
                    delta_pitch_str = "COUNTERRESET",
                    total_length=0,
                    resultPitch = "COUNTERRESET",
                    status = "COUNTERRESET",
                    NGreason = "COUNTERRESET")
        #check kouden sensor and tenmetsu status

        for i in range (len(self.inspection_config.kouden_sensor)):
            self.ethernet_status_red_tenmetsu_status[i] = self.inspection_config.kouden_sensor[i]
            self.ethernet_status_green_hold_status[i] = 0 #Reset at every loop
            self.ethernet_status_red_hold_status[i] = 0 #Reset at every loop

        for i in range(len(self.ethernet_status_red_tenmetsu_status)):
            if self.ethernet_status_red_tenmetsu_status[i] == 1:
                self.InspectionStatus[i] = "製品検出済み"
            else:
                self.InspectionStatus[i] = "製品未検出"

        if self.multiCam_stream is False:
            self.multiCam_stream = True
            self.initialize_all_camera()

        if not self.has_required_cameras([0, 1, 2, 3, 4, 5]):
            self.emit_blank_part_frames()
            self.emit_blank_hole_frames()
            self.msleep(100)
            return True

        self.bottomframe = self.read_camera_frame(self.cap_cam0)
        self.mergeframe1 = self.read_camera_frame(self.cap_cam1)
        self.mergeframe2 = self.read_camera_frame(self.cap_cam2)
        self.mergeframe3 = self.read_camera_frame(self.cap_cam3)
        self.mergeframe4 = self.read_camera_frame(self.cap_cam4)
        self.mergeframe5 = self.read_camera_frame(self.cap_cam5)

        if any(frame is None for frame in (self.bottomframe, self.mergeframe1, self.mergeframe2, self.mergeframe3, self.mergeframe4, self.mergeframe5)):
            self.emit_blank_part_frames()
            self.emit_blank_hole_frames()
            self.msleep(100)
            return True

        self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
        self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
        self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
        self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
        self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)

        #Downsampled the image
        self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
        self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)
        self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)
        self.mergeframe4_scaled = self.downSampling(self.mergeframe4, self.scaled_width, self.scaled_height)
        self.mergeframe5_scaled = self.downSampling(self.mergeframe5, self.scaled_width, self.scaled_height)

        if self.inspection_config.mapCalculated[1] is False:
            for i in range(1, 6):
                if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                    camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                    h, w = self.mergeframe1.shape[:2]
                    self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                    self.inspection_config.mapCalculated[i] = True
                    print(f"Calibration map is calculated for Camera {i}")

                    camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                    h, w = self.mergeframe1_scaled.shape[:2]
                    self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                    print(f"Calibration map is calculated for Camera {i} for scaled image")

        if self.inspection_config.mapCalculated[1] is True:

            #rotate the bottom frame 90deg CCW
            self.bottomframe = self.downScaledImage(self.bottomframe, self.scale_factor_hole)
            self.bottomframe = cv2.flip(cv2.rotate(self.bottomframe, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

            self.holeFrame1 = self.bottomframe[self.hole1Crop_XYpos_scaled[0]:self.hole1Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole1Crop_XYpos_scaled[1]:self.hole1Crop_XYpos_scaled[1] + self.width_hole_offset]
            self.holeFrame2 = self.bottomframe[self.hole2Crop_XYpos_scaled[0]:self.hole2Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole2Crop_XYpos_scaled[1]:self.hole2Crop_XYpos_scaled[1] + self.width_hole_offset]
            self.holeFrame3 = self.bottomframe[self.hole3Crop_XYpos_scaled[0]:self.hole3Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole3Crop_XYpos_scaled[1]:self.hole3Crop_XYpos_scaled[1] + self.width_hole_offset]
            self.holeFrame4 = self.bottomframe[self.hole4Crop_XYpos_scaled[0]:self.hole4Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole4Crop_XYpos_scaled[1]:self.hole4Crop_XYpos_scaled[1] + self.width_hole_offset]
            self.holeFrame5 = self.bottomframe[self.hole5Crop_XYpos_scaled[0]:self.hole5Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole5Crop_XYpos_scaled[1]:self.hole5Crop_XYpos_scaled[1] + self.width_hole_offset]

            if self.inspection_config.doInspection is False:

                self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.inspection_config.map1_downscaled[4], self.inspection_config.map2_downscaled[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.mergeframe5_scaled = cv2.remap(self.mergeframe5_scaled, self.inspection_config.map1_downscaled[5], self.inspection_config.map2_downscaled[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                self.combinedImage_scaled = self.homography_blank_canvas_scaled.copy()
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe1_scaled, "H1_scaled")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe2_scaled, "H2_scaled")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe3_scaled, "H3_scaled")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe4_scaled, "H4_scaled")
                self.combinedImage_scaled = self.merge_frame_with_cache(self.combinedImage_scaled, self.mergeframe5_scaled, "H5_scaled")

                self.combinedImage_scaled = self.apply_planarize_with_cache(self.combinedImage_scaled, "planarizeTransform_scaled", (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                # cv2.imwrite("combinedImage_scaled_inference.png", self.combinedImage_scaled)

                #Crop the image scaled for each part
                self.part1Crop_scaled = self.combinedImage_scaled[self.part1Crop_YPos_scaled : self.part1Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                self.part2Crop_scaled = self.combinedImage_scaled[self.part2Crop_YPos_scaled : self.part2Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                self.part3Crop_scaled = self.combinedImage_scaled[self.part3Crop_YPos_scaled : self.part3Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                self.part4Crop_scaled = self.combinedImage_scaled[self.part4Crop_YPos_scaled : self.part4Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                self.part5Crop_scaled = self.combinedImage_scaled[self.part5Crop_YPos_scaled : self.part5Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]

                self.part1Crop_scaled = self.downSampling(self.part1Crop_scaled, width=1771, height=24)
                self.part2Crop_scaled = self.downSampling(self.part2Crop_scaled, width=1771, height=24)
                self.part3Crop_scaled = self.downSampling(self.part3Crop_scaled, width=1771, height=24)
                self.part4Crop_scaled = self.downSampling(self.part4Crop_scaled, width=1771, height=24)
                self.part5Crop_scaled = self.downSampling(self.part5Crop_scaled, width=1771, height=24)

                self.holeImageMerge[0] = self.holeFrame1.copy()
                self.holeImageMerge[1] = self.holeFrame2.copy()
                self.holeImageMerge[2] = self.holeFrame3.copy()
                self.holeImageMerge[3] = self.holeFrame4.copy()
                self.holeImageMerge[4] = self.holeFrame5.copy()

                self.holeFrame1 = self.downScaledImage(self.holeFrame1, 1.5)
                self.holeFrame2 = self.downScaledImage(self.holeFrame2, 1.5)
                self.holeFrame3 = self.downScaledImage(self.holeFrame3, 1.5)
                self.holeFrame4 = self.downScaledImage(self.holeFrame4, 1.5)
                self.holeFrame5 = self.downScaledImage(self.holeFrame5, 1.5)

                for i in range(len(self.holeImageMerge)):
                    self.InspectionResult_HoleDetection[i] = self.hoodFR_holeDetectionModel(
                        cv2.cvtColor(self.holeImageMerge[i], cv2.COLOR_BGR2RGB),
                        stream=True, verbose=False, conf=0.3, imgsz=256
                    )
                    detectedid = []

                    for j, r in enumerate(self.InspectionResult_HoleDetection[i]):
                        for k, box in enumerate(r.boxes):
                            detectedid.append(box.cls.item())
                    # print(f"Detected ID[i]: {detectedid}")
                    if 0.0 in detectedid:
                        self.DetectionResult_HoleDetection[i] = 1
                    else:
                        self.DetectionResult_HoleDetection[i] = 0

                # print(self.DetectionResult_HoleDetection)

                self.InspectionResult_PitchMeasured = [None]*5
                self.InspectionResult_PitchResult = [None]*5

                if self.InspectionImages_prev[0] is not None and any(sensor != 0 for sensor in self.inspection_config.kouden_sensor):
                    #convert to bgr
                    self.part1Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[0], cv2.COLOR_RGB2BGR)
                    self.part2Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[1], cv2.COLOR_RGB2BGR)
                    self.part3Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[2], cv2.COLOR_RGB2BGR)
                    self.part4Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[3], cv2.COLOR_RGB2BGR)
                    self.part5Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[4], cv2.COLOR_RGB2BGR)

                    self.ethernet_status_red_tenmetsu_status = self.ethernet_status_red_tenmetsu_status_prev.copy()
                    self.ethernet_status_green_hold_status = self.ethernet_status_green_hold_status_prev.copy()
                    self.ethernet_status_red_hold_status = self.ethernet_status_red_hold_status_prev.copy()

                    self.InspectionResult_PitchMeasured = self.InspectionResult_PitchMeasured_prev.copy()
                    self.InspectionResult_PitchResult = self.InspectionResult_PitchResult_prev.copy()
                    self.InspectionStatus = self.InspectionStatus_prev.copy()

                    ng_exists = False
                    for status in self.InspectionResult_Status:
                        if status == "NG":
                            ng_exists = True
                            break

                    if ng_exists:
                        for i, sensor_value in enumerate(self.inspection_config.kouden_sensor):
                            if sensor_value == 0:  # 0 indicates the part has been removed
                                if self.InspectionResult_Status[i] == "OK":
                                    play_alarm_sound()  # Alarm if a good part is being removed
                                elif self.InspectionResult_Status[i] == "NG":
                                    self.InspectionResult_Status[i] = "None"
                                    play_picking_sound()  # Confirm removal of the NG part

                if all(sensor == 0 for sensor in self.inspection_config.kouden_sensor):
                    self.InspectionImages_prev[0] = None

                if self.part1Crop_scaled is not None:
                    self.part1Cam.emit(self.convertQImage(self.part1Crop_scaled))
                if self.part2Crop_scaled is not None:
                    self.part2Cam.emit(self.convertQImage(self.part2Crop_scaled))
                if self.part3Crop_scaled is not None:
                    self.part3Cam.emit(self.convertQImage(self.part3Crop_scaled))
                if self.part4Crop_scaled is not None:
                    self.part4Cam.emit(self.convertQImage(self.part4Crop_scaled))
                if self.part5Crop_scaled is not None:
                    self.part5Cam.emit(self.convertQImage(self.part5Crop_scaled))

                if self.holeFrame1 is not None:
                    self.hole1Cam.emit(self.convertQImage(self.holeFrame1))
                if self.holeFrame2 is not None:
                    self.hole2Cam.emit(self.convertQImage(self.holeFrame2))
                if self.holeFrame3 is not None:
                    self.hole3Cam.emit(self.convertQImage(self.holeFrame3))
                if self.holeFrame4 is not None:
                    self.hole4Cam.emit(self.convertQImage(self.holeFrame4))
                if self.holeFrame5 is not None:
                    self.hole5Cam.emit(self.convertQImage(self.holeFrame5))

                #Empty the Inspection Result

                self.hoodFR_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)

            if self.InspectionTimeStart is None:
                self.InspectionTimeStart = time.time()

            # print(time.time() - self.InspectionTimeStart)

            if self.firstTimeInspection is False:
                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

            # print(f"InspectionConfig: {self.inspection_config.doInspection}")

            if self.inspection_config.doInspection is True:
                if not self.has_required_models(8):
                    self.inspection_config.doInspection = False
                    self.emit_blank_part_frames()
                    self.emit_blank_hole_frames()
                    print("Skipping widget 8 inspection because required model weights are missing.")
                    return True

                self.inspection_config.doInspection = False

                if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                    #Break the bottom if
                    print("No Kensain Number Input")
                    for i in range (len(self.InspectionStatus)):
                        self.InspectionStatus[i] = "社員番号未入力"
                    self.hoodFR_InspectionStatus.emit(self.InspectionStatus)
                    return True

                if self.InspectionTimeStart is not None:

                    if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:
                        self.firstTimeInspection is False
                        print("Inspection Started")
                        self.InspectionTimeStart = time.time()

                        for i in range (len(self.InspectionStatus)):
                            self.InspectionStatus[i] = "検査中"
                        self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                        self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[2], self.inspection_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe4 = cv2.remap(self.mergeframe4, self.inspection_config.map1[4], self.inspection_config.map2[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe5 = cv2.remap(self.mergeframe5, self.inspection_config.map1[5], self.inspection_config.map2[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage = self.homography_blank_canvas.copy()
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe1, "H1")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe2, "H2")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe3, "H3")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe4, "H4")
                        self.combinedImage = self.merge_frame_with_cache(self.combinedImage, self.mergeframe5, "H5")

                        self.combinedImage = self.apply_planarize_with_cache(self.combinedImage, "planarizeTransform", (int(self.homography_size[1]), int(self.homography_size[0])))
                        self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/(1.26))))#1.48 for the qt, 1.26 for the aspect ratio

                        # Crop the image scaled for each part
                        self.part1Crop = self.combinedImage[int(self.part1Crop_YPos*1.48) : int((self.part1Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part2Crop = self.combinedImage[int(self.part2Crop_YPos*1.48) : int((self.part2Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part3Crop = self.combinedImage[int(self.part3Crop_YPos*1.48) : int((self.part3Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part4Crop = self.combinedImage[int(self.part4Crop_YPos*1.48) : int((self.part4Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                        self.part5Crop = self.combinedImage[int(self.part5Crop_YPos*1.48) : int((self.part5Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]

                        #Need to convert to BGR for SAHI Inspection
                        self.part1Crop = cv2.cvtColor(self.part1Crop, cv2.COLOR_RGB2BGR)
                        self.part2Crop = cv2.cvtColor(self.part2Crop, cv2.COLOR_RGB2BGR)
                        self.part3Crop = cv2.cvtColor(self.part3Crop, cv2.COLOR_RGB2BGR)
                        self.part4Crop = cv2.cvtColor(self.part4Crop, cv2.COLOR_RGB2BGR)
                        self.part5Crop = cv2.cvtColor(self.part5Crop, cv2.COLOR_RGB2BGR)

                        #Put the All the image into a list
                        self.InspectionImages[0] = self.part1Crop.copy()
                        self.InspectionImages[1] = self.part2Crop.copy()
                        self.InspectionImages[2] = self.part3Crop.copy()
                        self.InspectionImages[3] = self.part4Crop.copy()
                        self.InspectionImages[4] = self.part5Crop.copy()

                        print(f"Length of Inspection Images : {len(self.InspectionImages)}")

                        #Emit　検査中 to the status signal

                        # # Do the inspection
                        for i in range(len(self.InspectionImages)):
                            #Only do inspectino on the one with kouden sensor on
                            if self.inspection_config.kouden_sensor[i] == 1:
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                    self.InspectionImages[i],
                                    self.hoodFR_clipDetectionModel,
                                    slice_height=180, slice_width=1280,
                                    overlap_height_ratio=0.0, overlap_width_ratio=0.3,
                                    postprocess_match_metric="IOS",
                                    postprocess_match_threshold=0.005,
                                    postprocess_class_agnostic=True,
                                    postprocess_type="GREEDYNMM",
                                    verbose=0,
                                    perform_standard_pred=True
                                )

                                #Crop image from 0 to 1024 width for the left and width-1024 to 1024 for the right
                                self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1024, :]
                                self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1024:, :]

                                self.InspectionImages_endSegmentation_Left[i] = add_imageborder(img = self.InspectionImages_endSegmentation_Left[i], width = 480)
                                self.InspectionImages_endSegmentation_Right[i] = add_imageborder(img = self.InspectionImages_endSegmentation_Right[i], width = 480)

                                #bgr to rgb
                                self.InspectionImages_endSegmentation_Left[i] = cv2.cvtColor(self.InspectionImages_endSegmentation_Left[i], cv2.COLOR_BGR2RGB)
                                self.InspectionImages_endSegmentation_Right[i] = cv2.cvtColor(self.InspectionImages_endSegmentation_Right[i], cv2.COLOR_BGR2RGB)

                                # cv2.imwrite(f"InspectionImages_endSegmentation_Left_{i}.png", self.InspectionImages_endSegmentation_Left[i])
                                # cv2.imwrite(f"InspectionImages_endSegmentation_Right_{i}.png", self.InspectionImages_endSegmentation_Right[i])

                                self.InspectionResult_EndSegmentation_Left[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=1680, verbose=False, retina_masks=True)
                                self.InspectionResult_EndSegmentation_Right[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=1680, verbose=False, retina_masks=True)

                                self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = partcheck(self.InspectionImages[i],
                                                                                                                                                                                  self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                  self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                  self.InspectionResult_EndSegmentation_Right[i])
                            else:
                                #Make pure black image
                                self.InspectionImages[i] = np.full((24, 1771, 3), (10, 10, 20), dtype=np.uint8)
                                self.InspectionResult_PitchMeasured[i] = None
                                self.InspectionResult_PitchResult[i] = None
                                self.InspectionResult_DetectionID[i] = None
                                self.InspectionResult_Status[i] = None

                            # print(self.InspectionResult_Status[i])
                            # print(self.InspectionResult_DetectionID[i])

                            if self.InspectionResult_Status[i] == "OK":
                                self.ethernet_status_green_hold_status[i] = 1
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 0

                                self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1

                                #Play konpou sound if the current_numofPart is dividable by 25
                                if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 25 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                                    play_konpou_sound()

                                self.InspectionStatus[i] = "OK"

                            elif self.InspectionResult_Status[i] == "NG":
                                self.ethernet_status_green_hold_status[i] = 0
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 1
                                self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1

                                self.InspectionStatus[i] = "NG"

                            else:
                                self.ethernet_status_green_hold_status[i] = 0
                                self.ethernet_status_red_tenmetsu_status[i] = 0
                                self.ethernet_status_red_hold_status[i] = 0

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                                currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                deltaTime = 0.0,
                                kensainName = self.inspection_config.kensainNumber,
                                detected_pitch_str = self.InspectionResult_PitchMeasured[i],
                                delta_pitch_str = self.InspectionResult_DeltaPitch[i],
                                total_length=0,
                                resultPitch = self.InspectionResult_PitchResult[i],
                                status = self.InspectionResult_Status[i],
                                NGreason = self.InspectionResult_NGReason[i])


                            #save hole image
                            self.save_image_hole(self.holeFrame1, False, "P1")
                            self.save_image_hole(self.holeFrame2, False, "P2")
                            self.save_image_hole(self.holeFrame3, False, "P3")
                            self.save_image_hole(self.holeFrame4, False, "P4")
                            self.save_image_hole(self.holeFrame5, False, "P5")

                            self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                        self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                        self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                        self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                        self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                        self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                        self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1771, height=24)
                        self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1771, height=24)
                        self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1771, height=24)
                        self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1771, height=24)
                        self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1771, height=24)

                        self.hoodFR_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                        self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)
                        print("Inspection Finished")
                        #Remember that list is mutable
                        self.ethernet_status_red_tenmetsu_status_prev = self.ethernet_status_red_tenmetsu_status_prev.copy()
                        self.ethernet_status_green_hold_status_prev = self.ethernet_status_green_hold_status.copy()
                        self.ethernet_status_red_hold_status_prev = self.ethernet_status_red_hold_status.copy()

                        self.InspectionImages_prev[0] = self.InspectionImages[0]
                        self.InspectionImages_prev[1] = self.InspectionImages[1]
                        self.InspectionImages_prev[2] = self.InspectionImages[2]
                        self.InspectionImages_prev[3] = self.InspectionImages[3]
                        self.InspectionImages_prev[4] = self.InspectionImages[4]

                        self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                        self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()
                        self.InspectionStatus_prev = self.InspectionStatus.copy()

                        self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                        self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                        self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                        self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                        self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                        self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                        self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                        self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                        # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

        #emit the ethernet
        self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
        self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
        self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
        self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

        # Emit status based on the red tenmetsu status

        self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

        # Emit the hole detection
        self.hoodFR_HoleStatus.emit(self.DetectionResult_HoleDetection)
        return False


    def setCounterFalse(self):
        self.inspection_config.furyou_plus = False
        self.inspection_config.furyou_minus = False
        self.inspection_config.kansei_plus = False
        self.inspection_config.kansei_minus = False
        self.inspection_config.furyou_plus_10 = False
        self.inspection_config.furyou_minus_10 = False
        self.inspection_config.kansei_plus_10 = False
        self.inspection_config.kansei_minus_10 = False

    def manual_adjustment(self, currentPart, Totalpart,
                          furyou_plus, furyou_minus, 
                          furyou_plus_10, furyou_minus_10,
                          kansei_plus, kansei_minus,
                          kansei_plus_10, kansei_minus_10):
        
        ok_count_current = currentPart[0]
        ng_count_current = currentPart[1]
        ok_count_total = Totalpart[0]
        ng_count_total = Totalpart[1]
        
        if furyou_plus:
            ng_count_current += 1
            ng_count_total += 1

        if furyou_plus_10:
            ng_count_current += 10
            ng_count_total += 10

        if furyou_minus and ng_count_current > 0 and ng_count_total > 0:
            ng_count_current -= 1
            ng_count_total -= 1
        
        if furyou_minus_10 and ng_count_current > 9 and ng_count_total > 9:
            ng_count_current -= 10
            ng_count_total -= 10

        if kansei_plus:
            ok_count_current += 1
            ok_count_total += 1

        if kansei_plus_10:
            ok_count_current += 10
            ok_count_total += 10

        if kansei_minus and ok_count_current > 0 and ok_count_total > 0:
            ok_count_current -= 1
            ok_count_total -= 1

        if kansei_minus_10 and ok_count_current > 9 and ok_count_total > 9:
            ok_count_current -= 10
            ok_count_total -= 10

        self.setCounterFalse()

        self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                numofPart = [ok_count_total, ng_count_total], 
                currentnumofPart = [ok_count_current, ng_count_current],
                deltaTime = 0.0,
                kensainName = self.inspection_config.kensainNumber, 
                detected_pitch_str = "MANUAL", 
                delta_pitch_str = "MANUAL", 
                total_length=0,
                resultPitch = "MANUAL",
                status = "MANUAL",
                NGreason = "MANUAL")

        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    
    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, detected_pitch_str, 
                             delta_pitch_str, total_length, 
                             resultPitch, status, NGreason):
        # Ensure all inputs are strings or compatible types

        timestamp = datetime.now()
        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H:%M:%S")

        partname = str(partname)
        numofPart = str(numofPart)
        currentnumofPart = str(currentnumofPart)
        timestamp_hour = str(timestamp_hour)
        timestamp_date = str(timestamp_date)
        deltaTime = float(deltaTime)  # Ensure this is a float
        kensainName = str(kensainName)
        detected_pitch_str = str(detected_pitch_str)
        delta_pitch_str = str(delta_pitch_str)
        total_length = float(total_length)  # Ensure this is a float
        resultPitch = str(resultPitch)
        status = str(status)
        NGreason = str(NGreason)

        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason))
        self.conn.commit()

        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        #Also save to mysql cursor
        self.mysql_cursor.execute('''
        INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason))
        self.mysql_conn.commit()


    def get_last_entry_currentnumofPart(self, part_name):
        self.cursor.execute('''
        SELECT currentnumofPart 
        FROM inspection_results 
        WHERE partName = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name,))
        
        row = self.cursor.fetchone()
        if row:
            currentnumofPart = eval(row[0])
            return currentnumofPart
        else:
            return [0, 0]
            
    def get_last_entry_total_numofPart(self, part_name):
        # Get today's date in yyyymmdd format
        today_date = datetime.now().strftime("%Y%m%d")

        self.cursor.execute('''
        SELECT numofPart 
        FROM inspection_results 
        WHERE partName = ? AND timestampDate = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name, today_date))
        
        row = self.cursor.fetchone()
        if row:
            numofPart = eval(row[0])  # Convert the string tuple to an actual tuple
            return numofPart
        else:
            return [0, 0]  # Default values if no entry is found    def get_last_entry_currentnumofPart(self, part_name):
        self.cursor.execute('''
        SELECT currentnumofPart 
        FROM inspection_results 
        WHERE partName = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name,))
        
        row = self.cursor.fetchone()
        if row:
            currentnumofPart = eval(row[0])
            return currentnumofPart
        else:
            return [0, 0]
            

    def draw_status_text_PIL(self, image, text, color, size = "normal", x_offset = 0, y_offset = 0):

        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2

        if size == "large":
            font_scale = 130.0

        if size == "normal":
            font_scale = 100.0

        elif size == "small":
            font_scale = 50.0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.kanjiFontPath, font_scale)

        draw.text((center_x + x_offset, center_y + y_offset), text, font=font, fill=color)  
        # Convert back to BGR for OpenCV compatibility
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return image

    def save_image(self, image):
        dir = "aikensa/inspection/" + self.widget_dir_map[self.inspection_config.widget]
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image)

    def save_image_hole(self, image, BGR = True, id=None):
        if BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/hole/"
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id + ".png", image)


    def save_image_result(self, image_initial, image_result, result, BGR = True, id = None):
        if BGR:
            image_initial = cv2.cvtColor(image_initial, cv2.COLOR_RGB2BGR)
            image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)

        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id + ".png", image_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id + ".png", image_result)

    def minitimerStart(self):
        self.timerStart_mini = time.time()
    
    def minitimerFinish(self, message = "OperationName"):
        self.timerFinish_mini = time.time()
        # self.fps_mini = 1/(self.timerFinish_mini - self.timerStart_mini)
        print(f"Time to {message} : {(self.timerFinish_mini - self.timerStart_mini) * 1000} ms")
        # print(f"FPS of {message} : {self.fps_mini}")

    def convertQImage(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image
    
    def converQImageRGB(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image
    
    def downScaledImage(self, image, scaleFactor=1.0):
        #create a copy of the image
        resized_image = cv2.resize(image, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor, interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def downSampling(self, image, width=384, height=256):
        #create a copy of the image
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff

    def load_homography_matrix(self, filename):
        with open(filename, 'r') as file:
            homography_matrix = yaml.load(file, Loader=yaml.FullLoader)
        return np.array(homography_matrix, dtype=np.float64)

    def load_homography_adjustment_config(self, config_path):
        default_camera_adjustment = {
            "x_offset": 0.0,
            "y_offset": 0.0,
            "rotation_deg": 0.0,
        }
        default_config = {
            f"camera_{camera_index}": default_camera_adjustment.copy()
            for camera_index in range(1, 6)
        }

        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                loaded_config = yaml.load(file, Loader=yaml.FullLoader) or {}
            if isinstance(loaded_config, dict):
                if any(key in loaded_config for key in ("x_offset", "y_offset", "rotation_deg")):
                    legacy_adjustment = default_camera_adjustment.copy()
                    legacy_adjustment.update({
                        "x_offset": loaded_config.get("x_offset", 0.0),
                        "y_offset": loaded_config.get("y_offset", 0.0),
                        "rotation_deg": loaded_config.get("rotation_deg", 0.0),
                    })
                    default_config = {
                        f"camera_{camera_index}": legacy_adjustment.copy()
                        for camera_index in range(1, 6)
                    }
                else:
                    for camera_index in range(1, 6):
                        camera_key = f"camera_{camera_index}"
                        camera_adjustment = loaded_config.get(camera_key, {})
                        if isinstance(camera_adjustment, dict):
                            default_config[camera_key].update(camera_adjustment)

        return default_config

    def build_homography_adjustment_matrix(self, x_offset, y_offset, rotation_deg, image_size):
        height, width = image_size
        center_x = width / 2.0
        center_y = height / 2.0
        theta = np.deg2rad(rotation_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        translate_to_origin = np.array([
            [1.0, 0.0, -center_x],
            [0.0, 1.0, -center_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        translate_back = np.array([
            [1.0, 0.0, center_x],
            [0.0, 1.0, center_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        translation_matrix = np.array([
            [1.0, 0.0, x_offset],
            [0.0, 1.0, y_offset],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rotation_about_center = translate_back @ rotation_matrix @ translate_to_origin
        return translation_matrix @ rotation_about_center

    def apply_homography_adjustment(self, homography_matrix, adjustment_matrix):
        if homography_matrix is None:
            return None
        return adjustment_matrix @ homography_matrix

    def refresh_crop_position_cache(self):
        self.part1Crop_YPos_scaled = int(self.part1Crop_YPos // self.scale_factor)
        self.part2Crop_YPos_scaled = int(self.part2Crop_YPos // self.scale_factor)
        self.part3Crop_YPos_scaled = int(self.part3Crop_YPos // self.scale_factor)
        self.part4Crop_YPos_scaled = int(self.part4Crop_YPos // self.scale_factor)
        self.part5Crop_YPos_scaled = int(self.part5Crop_YPos // self.scale_factor)

        self.part1Crop_Ypos_hoodFR_scaled = int(self.part1Crop_YPos_hoodFR // self.scale_factor)
        self.part2Crop_Ypos_hoodFR_scaled = int(self.part2Crop_YPos_hoodFR // self.scale_factor)
        self.part3Crop_Ypos_hoodFR_scaled = int(self.part3Crop_YPos_hoodFR // self.scale_factor)
        self.part4Crop_Ypos_hoodFR_scaled = int(self.part4Crop_YPos_hoodFR // self.scale_factor)
        self.part5Crop_Ypos_hoodFR_scaled = int(self.part5Crop_YPos_hoodFR // self.scale_factor)

    def build_default_crop_settings(self):
        settings = {}
        for profile_key, profile_config in self.crop_position_profiles.items():
            settings[profile_key] = {
                f"part{index}": int(default_value)
                for index, default_value in enumerate(profile_config["defaults"], start=1)
            }
        return settings

    def get_current_crop_settings(self):
        settings = {}
        for profile_key, profile_config in self.crop_position_profiles.items():
            settings[profile_key] = {
                f"part{index}": int(getattr(self, attr_name))
                for index, attr_name in enumerate(profile_config["attrs"], start=1)
            }
        return settings

    def apply_crop_profile_settings(self, profile_key, profile_settings):
        profile_config = self.crop_position_profiles.get(profile_key)
        if profile_config is None:
            return

        for index, attr_name in enumerate(profile_config["attrs"], start=1):
            default_value = profile_config["defaults"][index - 1]
            setattr(self, attr_name, int(profile_settings.get(f"part{index}", default_value)))

        self.refresh_crop_position_cache()

    def save_crop_settings(self):
        os.makedirs(os.path.dirname(self.crop_settings_path), exist_ok=True)
        with open(self.crop_settings_path, "w") as file:
            yaml.dump(self.get_current_crop_settings(), file, sort_keys=False)

    def load_crop_settings(self):
        settings = self.build_default_crop_settings()

        if os.path.exists(self.crop_settings_path):
            with open(self.crop_settings_path, "r") as file:
                loaded_settings = yaml.load(file, Loader=yaml.FullLoader) or {}

            if isinstance(loaded_settings, dict):
                for profile_key, profile_settings in loaded_settings.items():
                    if profile_key in settings and isinstance(profile_settings, dict):
                        for part_key in settings[profile_key]:
                            if part_key in profile_settings:
                                settings[profile_key][part_key] = int(profile_settings[part_key])

        for profile_key, profile_settings in settings.items():
            self.apply_crop_profile_settings(profile_key, profile_settings)

        self.save_crop_settings()

    def clamp_crop_position(self, widget_index, position):
        position = max(0, int(position))

        if self.homography_size is None:
            return position

        crop_height = self.part_height_offset_nissanhoodFR if widget_index == 9 else self.part_height_offset
        visible_height = int(self.homography_size[0] / (1.26 * 1.48))
        max_position = max(0, visible_height - crop_height)
        return min(position, max_position)

    def adjust_crop_position(self, widget_index, part_index, delta):
        profile_key = self.widget_dir_map.get(widget_index)
        profile_config = self.crop_position_profiles.get(profile_key)

        if profile_config is None or not 1 <= part_index <= len(profile_config["attrs"]):
            return

        attr_name = profile_config["attrs"][part_index - 1]
        current_value = getattr(self, attr_name)
        new_value = self.clamp_crop_position(widget_index, current_value + int(delta))
        setattr(self, attr_name, new_value)

        self.refresh_crop_position_cache()
        self.save_crop_settings()
        self.emit_crop_values(widget_index)
        print(f"Updated {profile_key} part{part_index} crop Y to {new_value}")

    def get_crop_values_for_widget(self, widget_index):
        profile_key = self.widget_dir_map.get(widget_index)
        profile_config = self.crop_position_profiles.get(profile_key)

        if profile_config is None:
            return []

        return [int(getattr(self, attr_name)) for attr_name in profile_config["attrs"]]

    def emit_crop_values(self, widget_index=None):
        target_widgets = [widget_index] if widget_index is not None else list(self.widget_dir_map.keys())

        for current_widget_index in target_widgets:
            crop_values = self.get_crop_values_for_widget(current_widget_index)
            if crop_values:
                self.cropPositionChanged.emit(current_widget_index, crop_values)

    def get_camera_capture(self, camera_id):
        return getattr(self, f"cap_cam{camera_id}", None)

    def has_required_cameras(self, camera_ids):
        return all(self.get_camera_capture(camera_id) is not None for camera_id in camera_ids)

    def read_camera_frame(self, capture):
        if capture is None:
            return None

        success, frame = capture.read()
        if not success:
            return None

        return frame

    def has_required_models(self, widget_index):
        return self.widget_model_ready.get(widget_index, True)

    def create_blank_qimage(self, width, height):
        blank_image = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
        return self.convertQImage(blank_image)

    def emit_blank_part_frames(self):
        blank_image = self.create_blank_qimage(1771, 24)
        self.part1Cam.emit(blank_image)
        self.part2Cam.emit(blank_image)
        self.part3Cam.emit(blank_image)
        self.part4Cam.emit(blank_image)
        self.part5Cam.emit(blank_image)

    def emit_blank_hole_frames(self):
        blank_image = self.create_blank_qimage(self.width_hole_offset, self.height_hole_offset)
        self.hole1Cam.emit(blank_image)
        self.hole2Cam.emit(blank_image)
        self.hole3Cam.emit(blank_image)
        self.hole4Cam.emit(blank_image)
        self.hole5Cam.emit(blank_image)

    def emit_blank_dailytenken_frame(self):
        self.dailytenkenCam.emit(self.create_blank_qimage(1710, 198))

    def create_perspective_remap(self, transform, output_size):
        if transform is None:
            return None

        width, height = output_size
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        ones = np.ones_like(grid_x)
        destination_points = np.stack((grid_x, grid_y, ones), axis=-1).reshape(-1, 3).T

        try:
            inverse_transform = np.linalg.inv(transform)
        except np.linalg.LinAlgError:
            return None

        source_points = inverse_transform @ destination_points
        source_w = source_points[2]
        valid = np.abs(source_w) > 1e-8
        source_points[0, valid] /= source_w[valid]
        source_points[1, valid] /= source_w[valid]
        source_points[0, ~valid] = -1
        source_points[1, ~valid] = -1

        map_x = source_points[0].reshape(height, width).astype(np.float32)
        map_y = source_points[1].reshape(height, width).astype(np.float32)

        return map_x, map_y

    def cache_perspective_maps(self):
        self.homographyWarpCache = {}
        self.planarizeWarpCache = {}

        homography_output_sizes = {
            "": (self.homography_size[1], self.homography_size[0]),
            "_FR": (self.homography_size[1], self.homography_size[0]),
            "_scaled": (self.homography_size_scaled[1], self.homography_size_scaled[0]),
            "_scaled_FR": (self.homography_size_scaled[1], self.homography_size_scaled[0]),
        }

        for camera_index in range(1, 6):
            for suffix, output_size in homography_output_sizes.items():
                attribute_name = f"H{camera_index}{suffix}"
                transform = getattr(self, attribute_name, None)
                cached_maps = self.create_perspective_remap(transform, output_size)
                if cached_maps is not None:
                    self.homographyWarpCache[attribute_name] = cached_maps

        planarize_sizes = {
            "planarizeTransform": (self.homography_size[1], self.homography_size[0]),
            "planarizeTransform_temp": (self.homography_size[1], self.homography_size[0]),
            "planarizeTransform_scaled": (self.homography_size_scaled[1], self.homography_size_scaled[0]),
            "planarizeTransform_temp_scaled": (self.homography_size_scaled[1], self.homography_size_scaled[0]),
        }

        for attribute_name, output_size in planarize_sizes.items():
            transform = getattr(self, attribute_name, None)
            cached_maps = self.create_perspective_remap(transform, output_size)
            if cached_maps is not None:
                self.planarizeWarpCache[attribute_name] = cached_maps

    def warp_frame_with_cache(self, frame, transform_attribute, output_size):
        cached_maps = self.homographyWarpCache.get(transform_attribute)
        transform = getattr(self, transform_attribute, None)

        if cached_maps is not None:
            return cv2.remap(frame, cached_maps[0], cached_maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if transform is None:
            return None

        return cv2.warpPerspective(frame, transform, output_size)

    def merge_frame_with_cache(self, canvas, frame, transform_attribute):
        warped_frame = self.warp_frame_with_cache(frame, transform_attribute, (canvas.shape[1], canvas.shape[0]))
        if warped_frame is None:
            return canvas

        if warped_frame.ndim == 3:
            valid_mask = np.any(warped_frame > 0, axis=2)
        else:
            valid_mask = warped_frame > 0

        canvas[valid_mask] = warped_frame[valid_mask]
        return canvas

    def apply_planarize_with_cache(self, image, transform_attribute, output_size):
        cached_maps = self.planarizeWarpCache.get(transform_attribute)
        transform = getattr(self, transform_attribute, None)

        if cached_maps is not None:
            return cv2.remap(image, cached_maps[0], cached_maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if transform is None:
            return image

        return cv2.warpPerspective(image, transform, output_size)

    def load_homography_set(self, suffix=""):
        adjustment_config = None
        attribute_suffix = suffix if suffix else ""

        if suffix == "":
            adjustment_config = self.load_homography_adjustment_config(self.homography_adjustment_path)

        if suffix == "_FR":
            adjustment_config = self.load_homography_adjustment_config(self.homography_adjustment_fr_path)

        for camera_index in range(1, 6):
            matrix_path = f"./aikensa/cameracalibration/homography_param_cam{camera_index}{suffix}.yaml"
            if os.path.exists(matrix_path):
                matrix = self.load_homography_matrix(matrix_path)
                if adjustment_config is not None:
                    camera_adjustment = adjustment_config.get(f"camera_{camera_index}", {})
                    adjustment_matrix = self.build_homography_adjustment_matrix(
                        camera_adjustment.get("x_offset", 0.0),
                        camera_adjustment.get("y_offset", 0.0),
                        camera_adjustment.get("rotation_deg", 0.0),
                        self.homography_size,
                    )
                    matrix = self.apply_homography_adjustment(matrix, adjustment_matrix)
                    self.homography_adjustment_FR[camera_index] = adjustment_matrix
                setattr(self, f"H{camera_index}{attribute_suffix}", matrix)
                print(f"Loaded homography matrix for camera {camera_index}{suffix}")

            scaled_matrix_path = f"./aikensa/cameracalibration/homography_param_cam{camera_index}_scaled{suffix}.yaml"
            if os.path.exists(scaled_matrix_path):
                scaled_matrix = self.load_homography_matrix(scaled_matrix_path)
                if adjustment_config is not None:
                    camera_adjustment = adjustment_config.get(f"camera_{camera_index}", {})
                    scaled_adjustment_matrix = self.build_homography_adjustment_matrix(
                        camera_adjustment.get("x_offset", 0.0) / self.scale_factor,
                        camera_adjustment.get("y_offset", 0.0) / self.scale_factor,
                        camera_adjustment.get("rotation_deg", 0.0),
                        self.homography_size_scaled,
                    )
                    scaled_matrix = self.apply_homography_adjustment(scaled_matrix, scaled_adjustment_matrix)
                setattr(self, f"H{camera_index}_scaled{attribute_suffix}", scaled_matrix)
                print(f"Loaded scaled homography matrix for camera {camera_index}{suffix}")

    def initialize_model(self):
        #Change based on the widget
        hoodFR_holeDetectionModel = None
        hoodFR_clipDetectionModel = None
        hoodFR_hanireDetectionModel = None
        hoodFR_endSegmentationModel = None
        P658207YA0A_clipDetectionModel = None

        #Classification Model
        path_hoodFR_clipDetectionModel = "./aikensa/models/65820W030P_CLIP.pt"
        path_hoodFR_holeDetectionModel = "./aikensa/models/65820W030P_MIZUANA.pt"
        path_P658207YA0A_clipDetectionModel = "./aikensa/models/658207YA0A_CLIP.pt"
        #Segmentation Model
        path_hoodFR_endSegmentationModel = "./aikensa/models/65820W030P_END_SEGMENTATION.pt"


        if os.path.exists(path_hoodFR_holeDetectionModel):
            try:
                hoodFR_holeDetectionModel = YOLO(path_hoodFR_holeDetectionModel)
            except Exception as error:
                print(f"Failed to load hole detection model: {error}")

        if os.path.exists(path_hoodFR_clipDetectionModel):
            try:
                hoodFR_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                                model_path=path_hoodFR_clipDetectionModel,
                                                                                confidence_threshold=0.7,
                                                                                device="cuda:0")
            except Exception as error:
                print(f"Failed to load 65820W030P clip model: {error}")

        if os.path.exists(path_hoodFR_endSegmentationModel):
            try:
                hoodFR_endSegmentationModel = YOLO(path_hoodFR_endSegmentationModel)
            except Exception as error:
                print(f"Failed to load end segmentation model: {error}")

        if os.path.exists(path_P658207YA0A_clipDetectionModel):
            try:
                P658207YA0A_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                                    model_path=path_P658207YA0A_clipDetectionModel,
                                                                                    confidence_threshold=0.5,
                                                                                    device="cuda:0")
            except Exception as error:
                print(f"Failed to load 658207YA0A clip model: {error}")

        self.hoodFR_holeDetectionModel = hoodFR_holeDetectionModel
        self.hoodFR_clipDetectionModel = hoodFR_clipDetectionModel
        self.hoodFR_hanireDetectionModel = hoodFR_hanireDetectionModel
        self.hoodFR_endSegmentationModel = hoodFR_endSegmentationModel

        self.P658207YA0A_clipDetectionModel = P658207YA0A_clipDetectionModel
        self.widget_model_ready = {
            8: all(model is not None for model in (
                self.hoodFR_holeDetectionModel,
                self.hoodFR_clipDetectionModel,
                self.hoodFR_endSegmentationModel,
            )),
            9: self.P658207YA0A_clipDetectionModel is not None,
            21: all(model is not None for model in (
                self.hoodFR_clipDetectionModel,
                self.hoodFR_endSegmentationModel,
            )),
        }


    def stop(self):
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        self.running = False
        print("Inspection thread stopped.")

    def add_columns(self, cursor, table_name, columns):
        for column_name, column_type in columns:
            try:
                cursor.execute(f'''
                ALTER TABLE {table_name}
                ADD COLUMN {column_name} {column_type};
                ''')
                print(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                print(f"Could not add column {column_name}: {e}")

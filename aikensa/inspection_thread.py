import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging

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
# from aikensa.parts_config.ctrplr_8283XW0W0P import partcheck as ctrplrCheck
# from aikensa.parts_config.ctrplr_8283XW0W0P import dailytenkencheck

from PIL import ImageFont, ImageDraw, Image

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

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

    def __init__(self, inspection_config: InspectionConfig = None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        
        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.cap_cam = None
        self.cap_cam0 = None
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None
        self.cap_cam4 = None
        self.cap_cam5 = None

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


        self.homography_size = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.scale_factor = 5.0
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.planarizeTransform = None
        self.planarizeTransform_scaled = None

    def release_all_camera(self):
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

        #print thread started
        print("Inspection Thread Started")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)


        #INIT all variables

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                self.H3 = np.array(self.homography_matrix3)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                self.H4 = np.array(self.homography_matrix4)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5.yaml") as file:
                self.homography_matrix5 = yaml.load(file, Loader=yaml.FullLoader)
                self.H5 = np.array(self.homography_matrix5)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml") as file:
                self.homography_matrix3_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H3_scaled = np.array(self.homography_matrix3_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml") as file:
                self.homography_matrix4_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H4_scaled = np.array(self.homography_matrix4_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml") as file:
                self.homography_matrix5_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H5_scaled = np.array(self.homography_matrix5_scaled)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_scaled = np.array(transform_list)

        while self.running:
                
            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget == 0:
                self.initialize_all_camera()

            _, self.mergeframe1 = self.cap_cam1.read()
            _, self.mergeframe2 = self.cap_cam2.read()
            _, self.mergeframe3 = self.cap_cam3.read()
            _, self.mergeframe4 = self.cap_cam4.read()
            _, self.mergeframe5 = self.cap_cam5.read()
            
            
        self.msleep(5)
            
    # print(f"Camera {self.inspection_config.cameraID} released.")

    def stop(self):
        self.running = False
        self.release_all_camera()
        print("Inspection thread stopped.")
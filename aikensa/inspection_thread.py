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
from aikensa.parts_config.hoodFR_65820W030P import partcheck

from PIL import ImageFont, ImageDraw, Image

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
        
        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        self.part_height_offset = 110
        self.part_height_offset_scaled = int(self.part_height_offset//self.scale_factor)

        self.part1Crop_YPos = 15
        self.part2Crop_YPos = 240
        self.part3Crop_YPos = 480
        self.part4Crop_YPos = 720
        self.part5Crop_YPos = 955

        self.part1Crop_YPos_scaled = int(self.part1Crop_YPos//self.scale_factor)
        self.part2Crop_YPos_scaled = int(self.part2Crop_YPos//self.scale_factor)
        self.part3Crop_YPos_scaled = int(self.part3Crop_YPos//self.scale_factor)
        self.part4Crop_YPos_scaled = int(self.part4Crop_YPos//self.scale_factor)
        self.part5Crop_YPos_scaled = int(self.part5Crop_YPos//self.scale_factor)

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

        self.InspectionResult_ClipDetection = [None]*5
        self.InspectionResult_PitchDetection = [None]*5
        self.InspectionResult_DeltaDetection = [None]*5

        self.DetectionResult_HoleDetection = [None]*5


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

        #print thread started
        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Model Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)


        #INIT all variables

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)
                print(f"Loaded homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)
                print(f"Loaded homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                self.H3 = np.array(self.homography_matrix3)
                print(f"Loaded homography matrix for camera 3")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                self.H4 = np.array(self.homography_matrix4)
                print(f"Loaded homography matrix for camera 4")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5.yaml") as file:
                self.homography_matrix5 = yaml.load(file, Loader=yaml.FullLoader)
                self.H5 = np.array(self.homography_matrix5)
                print(f"Loaded homography matrix for camera 5")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)
                print(f"Loaded scaled homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)
                print(f"Loaded scaled homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml") as file:
                self.homography_matrix3_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H3_scaled = np.array(self.homography_matrix3_scaled)
                print(f"Loaded scaled homography matrix for camera 3")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml") as file:
                self.homography_matrix4_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H4_scaled = np.array(self.homography_matrix4_scaled)
                print(f"Loaded scaled homography matrix for camera 4")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml") as file:
                self.homography_matrix5_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H5_scaled = np.array(self.homography_matrix5_scaled)
                print(f"Loaded scaled homography matrix for camera 5")

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

        while self.running:

                
            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget == 8:
                self.timerStart = time.time()


                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()

                _, self.bottomframe = self.cap_cam0.read()
                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()
                _, self.mergeframe5 = self.cap_cam5.read()

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


                    #rotate the bottom frame 90deg CCW
                    self.bottomframe = self.downScaledImage(self.bottomframe, self.scale_factor_hole)
                    self.bottomframe = cv2.flip(cv2.rotate(self.bottomframe, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

                    self.holeFrame1 = self.bottomframe[self.hole1Crop_XYpos_scaled[0]:self.hole1Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole1Crop_XYpos_scaled[1]:self.hole1Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame2 = self.bottomframe[self.hole2Crop_XYpos_scaled[0]:self.hole2Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole2Crop_XYpos_scaled[1]:self.hole2Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame3 = self.bottomframe[self.hole3Crop_XYpos_scaled[0]:self.hole3Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole3Crop_XYpos_scaled[1]:self.hole3Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame4 = self.bottomframe[self.hole4Crop_XYpos_scaled[0]:self.hole4Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole4Crop_XYpos_scaled[1]:self.hole4Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame5 = self.bottomframe[self.hole5Crop_XYpos_scaled[0]:self.hole5Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole5Crop_XYpos_scaled[1]:self.hole5Crop_XYpos_scaled[1] + self.width_hole_offset]
                
                    if self.inspection_config.doInspection is False:
                        self.mergeframe1_scaled = cv2.rotate(self.mergeframe1_scaled, cv2.ROTATE_180)
                        self.mergeframe2_scaled = cv2.rotate(self.mergeframe2_scaled, cv2.ROTATE_180)
                        self.mergeframe3_scaled = cv2.rotate(self.mergeframe3_scaled, cv2.ROTATE_180)
                        self.mergeframe4_scaled = cv2.rotate(self.mergeframe4_scaled, cv2.ROTATE_180)
                        self.mergeframe5_scaled = cv2.rotate(self.mergeframe5_scaled, cv2.ROTATE_180)

                        self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.inspection_config.map1_downscaled[4], self.inspection_config.map2_downscaled[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe5_scaled = cv2.remap(self.mergeframe5_scaled, self.inspection_config.map1_downscaled[5], self.inspection_config.map2_downscaled[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe3_scaled, self.H3_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe4_scaled, self.H4_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe5_scaled, self.H5_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                        self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

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

                        self.holeFrame1 = self.downScaledImage(self.holeFrame1, 2.0)
                        self.holeFrame2 = self.downScaledImage(self.holeFrame2, 2.0)
                        self.holeFrame3 = self.downScaledImage(self.holeFrame3, 2.0)
                        self.holeFrame4 = self.downScaledImage(self.holeFrame4, 2.0)
                        self.holeFrame5 = self.downScaledImage(self.holeFrame5, 2.0)
                        
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
  
                    if self.inspection_config.doInspection is True:
                        self.inspection_config.doInspection = False
                        print("Inspection Started") 

                        self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                        self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
                        self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                        self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
                        self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)

                        # cv2.imwrite("./mergeframe1.png", self.mergeframe1)
                        # cv2.imwrite("./mergeframe2.png", self.mergeframe2)
                        # cv2.imwrite("./mergeframe3.png", self.mergeframe3)
                        # cv2.imwrite("./mergeframe4.png", self.mergeframe4)
                        # cv2.imwrite("./mergeframe5.png", self.mergeframe5)

                        self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[2], self.inspection_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe4 = cv2.remap(self.mergeframe4, self.inspection_config.map1[4], self.inspection_config.map2[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe5 = cv2.remap(self.mergeframe5, self.inspection_config.map1[5], self.inspection_config.map2[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                        self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                        self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe3, self.H3)
                        self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe4, self.H4)
                        self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe5, self.H5)
                                                                    
                        self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform, (int(self.homography_size[1]), int(self.homography_size[0])))
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

                        #Put the All the image into a list
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        cv2.imwrite("./part1Crop.png", self.part1Crop)
                        cv2.imwrite("./part2Crop.png", self.part2Crop)
                        cv2.imwrite("./part3Crop.png", self.part3Crop)
                        cv2.imwrite("./part4Crop.png", self.part4Crop)
                        cv2.imwrite("./part5Crop.png", self.part5Crop)

                        print(f"Lengt of Inspection Images : {len(self.InspectionImages)}") 

                        # Do the inspection
                        for i in range(len(self.InspectionImages)):
                            print(f"Inspection Image {i}")
                            self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                self.InspectionImages[i], 
                                self.hoodFR_clipDetectionModel, 
                                slice_height=200, slice_width=968, 
                                overlap_height_ratio=0.3, overlap_width_ratio=0.2,
                                postprocess_match_metric="IOS",
                                postprocess_match_threshold=0.2,
                                postprocess_class_agnostic=True,
                                postprocess_type="GREEDYNMM",
                                verbose=0,
                                perform_standard_pred=False
                            )
                            print(f"Clip Detection Result : {self.InspectionResult_ClipDetection[i]}")

                            self.InspectionImages[i], self.InspectionResult_PitchDetection[i], self.InspectionResult_DeltaDetection[i] = partcheck(self.InspectionImages[i], self.InspectionResult_ClipDetection[i].object_prediction_list)

                        print("Inspection Finished")

        self.msleep(1)

    def minitimerStart(self):
        self.timerStart_mini = time.time()
    
    def minitimerFinish(self, message = "OperationName"):
        self.timerFinish_mini = time.time()
        # self.fps_mini = 1/(self.timerFinish_mini - self.timerStart_mini)
        print(f"Time to {message} : {(self.timerFinish_mini - self.timerStart_mini) * 1000} ms")
        # print(f"FPS of {message} : {self.fps_mini}")

    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
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

    def initialize_model(self):
        #Change based on the widget
        hoodFR_holeIdentification = None
        hoodFR_clipDetectionModel = None
        hoodFR_hanireDetectionModel = None

        #Detection Model
        path_hoodFR_holeIdentification = "./aikensa/models/classify_holes_or_not.pt"
        #Classification Model
        path_hoodFR_clipDetectionModel = "./aikensa/models/detect_clip_and_holes.pt"
        path_hoodFR_hanireDetectionModel = "./aikensa/models/classify_hanire.pt"

        if os.path.exists(path_hoodFR_holeIdentification):
            hoodFR_holeIdentification = YOLO(path_hoodFR_holeIdentification)
        
        if os.path.exists(path_hoodFR_clipDetectionModel):
            hoodFR_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                            model_path=path_hoodFR_clipDetectionModel,
                                                                            confidence_threshold=0.5,
                                                                            device="cuda:0")
        if os.path.exists(path_hoodFR_hanireDetectionModel):
            hoodFR_hanireDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                            model_path=path_hoodFR_hanireDetectionModel,
                                                                            confidence_threshold=0.6,
                                                                            device="cuda:0")

        self.hoodFR_holeIdentification = hoodFR_holeIdentification
        self.hoodFR_clipDetectionModel = hoodFR_clipDetectionModel
        self.hoodFR_hanireDetectionModel = hoodFR_hanireDetectionModel

        if self.hoodFR_holeIdentification is not None:
            print("HoodFR Hole Identification Model Initialized")
        if self.hoodFR_clipDetectionModel is not None:
            print("HoodFR Clip Detection Model Initialized")
        if self.hoodFR_hanireDetectionModel is not None:
            print("HoodFR Hanire Detection Model Initialized")






    def stop(self):
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        self.running = False
        print("Inspection thread stopped.")
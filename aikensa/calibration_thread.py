import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, calculateHomography, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize

from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound


@dataclass
class CalibrationConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected
    
    calculateSingeFrameMatrix: bool = False
    calculateCamMatrix: bool = False
    delCamMatrix: bool = False
    savecalculatedCamImage: bool = False

    calculateHomo_cam1: bool = False
    calculateHomo_cam2: bool = False
    calculateHomo_cam3: bool = False
    calculateHomo_cam4: bool = False
    calculateHomo_cam5: bool = False

    deleteHomo: bool = False

    mergeCam: bool = False
    saveImage: bool = False

    savePlanarize: bool = False
    delPlanarize: bool = False

    opacity: float = 0.5
    blur: int = 10
    lower_canny: int = 100
    upper_canny: int = 200
    contrast: int = 200
    brightness: int = 0
    savecannyparams: bool = False

    HDRes: bool = False


class CalibrationThread(QThread):

    CalibCamStream = pyqtSignal(QImage)

    def __init__(self, calib_config: CalibrationConfig = None):
        super(CalibrationThread, self).__init__()
        self.running = True
        self.charucoTimer = None
        self.kensatimer = None

        if calib_config is None:
            self.calib_config = CalibrationConfig()    
        else:
            self.calib_config = calib_config

        self.widget_dir_map={
            1: "CamCalibration1",
            2: "CamCalibration2",
            3: "CamCalibration3",
            4: "CamCalibration4",
            5: "CamCalibration5",
        }


        self.cameraMatrix = None
        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"
        self.cap_cam = None
        self.frame  = None

    def initialize_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            self.frame = self.create_placeholder_image()
        else:
            self.cap_cam = initialize_camera(camID)
            if not self.cap_cam.isOpened():
                print(f"Failed to open camera with ID {camID}")
                self.cap_cam = None
            else:
                print(f"Initialized Camera on ID {camID}")
            
    def run(self):

        self.current_cameraID = self.calib_config.cameraID
        self.initialize_camera(self.current_cameraID)

        
        while self.running:

            if self.calib_config.cameraID != self.current_cameraID:
                # Camera ID has changed, reinitialize the camera
                self.current_cameraID = self.calib_config.cameraID
                self.initialize_camera(self.current_cameraID)

            if self.cap_cam is not None:
                try:
                    ret, self.frame = self.cap_cam.read()
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    if not ret:
                        print("Failed to capture frame")
                        continue
                except cv2.error as e:
                    print("An error occurred while reading frames from the cameras:", str(e))

            if self.frame is None:
                # Display a placeholder image if no frame is captured
                self.frame = self.create_placeholder_image()

            else:
                self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

            if self.calib_config.widget == 0:
                self.calib_config.cameraID = -1

            if self.calib_config.widget == 1:
                self.calib_config.cameraID = 1
                #downsample to 1229 819
                if self.calib_config.calculateSingeFrameMatrix:
                    self.frame, _, _ = detectCharucoBoard(self.frame)

                    self.calib_config.calculateSingeFrameMatrix = False

                self.frame = self.downSampling(self.frame, 1229, 819)

            if self.calib_config.widget == 2:
                self.calib_config.cameraID = 2
                self.frame = self.downSampling(self.frame, 1229, 819)

                

            if self.calib_config.widget == 3:
                self.calib_config.cameraID = 3
                self.frame = self.downSampling(self.frame, 1229, 819)

            if self.calib_config.widget == 4:
                self.calib_config.cameraID = 4
                self.frame = self.downSampling(self.frame, 1229, 819)

            if self.calib_config.widget == 5:
                self.calib_config.cameraID = 5
                self.frame = self.downSampling(self.frame, 1229, 819)
            
            self.CalibCamStream.emit(self.convertQImage(self.frame))
            #check self.frame type and size
            print(f"Frame type: {type(self.frame)}, Frame size: {self.frame.shape}")

                
        print(f"Camera {self.calib_config.cameraID} released.")

    def create_placeholder_image(self):
        # Create a small black image with a white dot in the center
        size = 100
        placeholder = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(placeholder, (size // 2, size // 2), 10, (255, 255, 255), -1)
        return placeholder

    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def undistortFrame(self, frame,cameraMatrix, distortionCoeff):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.undistort(frame, cameraMatrix, distortionCoeff, None, cameraMatrix)
        return frame

    def stop(self):
        self.running = False
        time.sleep(0.5)
    
    def stop(self):
        self.running = False
        print(f"Running is set to {self.running}")
        if self.cap_cam is not None:
            self.cap_cam.release()

  
    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def downSampling(self, image, width=384, height=256):
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

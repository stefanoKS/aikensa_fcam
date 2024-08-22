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
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix, calculateHomography, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize

from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound


@dataclass
class CalibrationConfig:
    widget: int = 0
    cameraID: int = 0
    
    calculateCamMatrix: bool = False
    captureCam: bool = False
    delCamMatrix: bool = False
    checkUndistort: bool = False

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

    camStream = pyqtSignal(QImage)

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


    def run(self):

        # Initialize Camera
        cap_cam = initialize_camera(self.calib_config.cameraID)
        print(f"Initiliazing Camera 1.... Located on {self.calib_config.cameraID}")

        while self.running is True:

            try:
                ret, frame = cap_cam.read()
                
            except cv2.error as e:
                print("An error occurred while reading frames from the cameras:", str(e))

            if frame is None:
                frame = np.zeros((2048, 3072, 3), dtype=np.uint8)

            else:
                frame = cv2.rotate(frame1, cv2.ROTATE_180)
                
                if os.path.exists("./aikensa/cameracalibration/cam1calibration_param.yaml"):
                    with open("./aikensa/cameracalibration/cam1calibration_param.yaml") as file:
                        cam1calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                        cameraMatrix1 = np.array(cam1calibration_param.get('camera_matrix'))
                        distortionCoeff1 = np.array(cam1calibration_param.get('distortion_coefficients'))

                    frame1raw = frame1.copy()
                    frame1 = cv2.undistort(frame1, cameraMatrix1, distortionCoeff1, None, cameraMatrix1)

                    if self.cam_config.checkUndistort1 == True:
                        cv2.imwrite("camu1ndistorted.jpg", frame1)
                        cv2.imwrite("cam1raw.jpg", frame1raw)
                        self.cam_config.checkUndistort1 = False

                if os.path.exists("./aikensa/cameracalibration/cam2calibration_param.yaml"):
                    with open("./aikensa/cameracalibration/cam2calibration_param.yaml") as file:
                        cam2calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                        cameraMatrix2 = np.array(cam2calibration_param.get('camera_matrix'))
                        distortionCoeff2 = np.array(cam2calibration_param.get('distortion_coefficients'))

                        frame2raw = frame2.copy()
                        frame2 = cv2.undistort(frame2, cameraMatrix2, distortionCoeff2, None, cameraMatrix2)
                        if self.cam_config.checkUndistort2 == True:
                            cv2.imwrite("camu2ndistorted.jpg", frame2)
                            cv2.imwrite("cam2raw.jpg", frame2raw)
                            self.cam_config.checkUndistort2 = False

                if self.cam_config.delCamMatrix1 == True:
                    if os.path.exists("./aikensa/cameracalibration/cam1calibration_param.yaml"):
                        os.remove("./aikensa/cameracalibration/cam1calibration_param.yaml")
                    self.cam_config.delCamMatrix1 = False

                if self.cam_config.delCamMatrix2 == True:
                    if os.path.exists("./aikensa/cameracalibration/cam2calibration_param.yaml"):
                        os.remove("./aikensa/cameracalibration/cam2calibration_param.yaml")
                    self.cam_config.delCamMatrix2 = False
                
                if ret1 and ret2:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    image1 = self.downSampling(frame1)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    image2 = self.downSampling(frame2)

                if self.cam_config.captureCam1 == True:
                    frame1, _, _ = detectCharucoBoard(frame1)

                    arucoFrame1 = frame1.copy()
                    self.charucoTimer = current_time

                    self.cam_config.captureCam1 = False

                    if self.charucoTimer and current_time - self.charucoTimer < 1:
                        image1 = self.downSampling(arucoFrame1)
                    elif self.charucoTimer and current_time - self.charucoTimer >= 1.2:
                        self.charucoTimer = None
                
                if self.cam_config.captureCam2 == True:
                    frame2, _, _ = detectCharucoBoard(frame2)

                    arucoFrame2 = frame2.copy()
                    self.charucoTimer = current_time

                    self.cam_config.captureCam2 = False

                    if self.charucoTimer and current_time - self.charucoTimer > 1:
                        image2 = self.downSampling(arucoFrame2)
                    else:
                        self.charucoTimer = None

                if self.cam_config.calculateCamMatrix1 == True:
                    calibration_matrix = calculatecameramatrix()
                    if not os.path.exists("./aikensa/cameracalibration"):
                        os.makedirs("./aikensa/cameracalibration")
                    with open("./aikensa/cameracalibration/cam1calibration_param.yaml", "w") as file:
                        yaml.dump(calibration_matrix, file)


                    print("Camera matrix 1 calculated.")
                    self.cam_config.calculateCamMatrix1 = False

                if self.cam_config.calculateCamMatrix2 == True:
                    calibration_matrix = calculatecameramatrix()
                    if not os.path.exists("./aikensa/cameracalibration"):
                        os.makedirs("./aikensa/cameracalibration")
                    with open("./aikensa/cameracalibration/cam2calibration_param.yaml", "w") as file:
                        yaml.dump(calibration_matrix, file)


                    print("Camera matrix 2 calculated.")
                    
                    self.cam_config.calculateCamMatrix2 = False

                if self.cam_config.calculateHomo == True:
                    combineImage, homographyMatrix = calculateHomography(frame1, frame2)
                    combineImage_lowres, homographyMatrix_lowres = calculateHomography(self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)),self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)))

                    os.makedirs("./aikensa/cameracalibration", exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param.yaml", "w") as file:
                        yaml.dump(homographyMatrix.tolist(), file)
                    with open("./aikensa/cameracalibration/homography_param_lowres.yaml", "w") as file:
                        yaml.dump(homographyMatrix_lowres.tolist(), file)
                    
                    self.cam_config.calculateHomo = False


                if self.cam_config.calculateHomo_cam1 == True:
                    _, homographyMatrix1 = calculateHomography_template(homography_template, frame1)
                    _, homographyMatrix1_lowres = calculateHomography_template(self.resizeImage(homography_template, int(homography_template.shape[1]//self.scale_factor), int(homography_template.shape[0]//self.scale_factor)),
                                                                                self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)))

                    print(f"Homography Matrix 1: {homographyMatrix1}")
                    print(f"Homography Matrix 1 Lowres: {homographyMatrix1_lowres}")

                    os.makedirs("./aikensa/cameracalibration", exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param_cam1.yaml", "w") as file:
                        yaml.dump(homographyMatrix1.tolist(), file)
                    with open("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml", "w") as file:
                        yaml.dump(homographyMatrix1_lowres.tolist(), file)

                    self.cam_config.calculateHomo_cam1 = False

                if self.cam_config.calculateHomo_cam2 == True:
                    _, homographyMatrix2 = calculateHomography_template(homography_template, frame2)
                    _, homographyMatrix2_lowres = calculateHomography_template(self.resizeImage(homography_template, int(homography_template.shape[1]//self.scale_factor), int(homography_template.shape[0]//self.scale_factor)),
                                                                                self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)))

                    print(f"Homography Matrix 2: {homographyMatrix2}")
                    print(f"Homography Matrix 2 Lowres: {homographyMatrix2_lowres}")

                    os.makedirs("./aikensa/cameracalibration", exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param_cam2.yaml", "w") as file:
                        yaml.dump(homographyMatrix2.tolist(), file)
                    with open("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml", "w") as file:
                        yaml.dump(homographyMatrix2_lowres.tolist(), file)

                    self.cam_config.calculateHomo_cam2 = False

                

                if self.cam_config.deleteHomo == True:
                    if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
                        os.remove("./aikensa/cameracalibration/homography_param.yaml")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_lowres.yaml"):
                        os.remove("./aikensa/cameracalibration/homography_param_lowres.yaml")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
                        os.remove("./aikensa/cameracalibration/homography_param_cam1.yaml")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
                        os.remove("./aikensa/cameracalibration/homography_param_cam2.yaml")
                    self.cam_config.deleteHomo = False



                # #blank combinedImage (placeholder)
                # combinedImage = homography_blank_canvas.copy()

                if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
                    with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                        homography_param1 = yaml.load(file, Loader=yaml.FullLoader)
                        H1 = np.array(homography_param1)
                        combinedImage = warpTwoImages_template(homography_blank_canvas, frame1, H1)

                        # cv2.imwrite("combinedImage_cam1.jpg", combinedImage)

                if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
                    with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                        homography_param2 = yaml.load(file, Loader=yaml.FullLoader)
                        H2 = np.array(homography_param2)
                        combinedImage = warpTwoImages_template(combinedImage, frame2, H2)

                        # cv2.imwrite("combinedImage_cam2.jpg", combinedImage)

                
                if os.path.exists("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml") and os.path.exists("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml"):
                    with open("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml") as file:
                        homography_param_lowres_cam1 = yaml.load(file, Loader=yaml.FullLoader)
                        H1_lowres = np.array(homography_param_lowres_cam1)
                        combinedImage_lowres = warpTwoImages_template(self.resizeImage(homography_blank_canvas, int(homography_blank_canvas.shape[1]//self.scale_factor), int(homography_blank_canvas.shape[0]//self.scale_factor)),
                                                                                self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)), 
                                                                                H1_lowres)
                        
                        # cv2.imwrite("combinedImage_lowres_cam1.jpg", combinedImage_lowres)

                    with open("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml") as file:
                        homography_param_lowres_cam2 = yaml.load(file, Loader=yaml.FullLoader)
                        H2_lowres = np.array(homography_param_lowres_cam2)
                        combinedImage_lowres = warpTwoImages_template(combinedImage_lowres, 
                                                                        self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)), H2_lowres)

                        # cv2.imwrite("combinedImage_lowres_cam2.jpg", combinedImage_lowres)

                combinedImage, _ = planarize(combinedImage, scale_factor=1.0)
                combinedImage_lowres, _t = planarize(combinedImage_lowres, self.scale_factor)

                # print("warp Transform;", _)
                # print("warp Transform Lowres;", _t)

                # cv2.imwrite("combinedImage.jpg", combinedImage)
                # cv2.imwrite("combinedImage_lowres.jpg", combinedImage_lowres)

                if self.cam_config.savePlanarize == True:
                    os.makedirs("./aikensa/param", exist_ok=True)
                    with open('./aikensa/param/warptransform.yaml', 'w') as file:
                        yaml.dump(_.tolist(), file)
                    with open('./aikensa/param/warptransform_lowres.yaml', 'w') as file:
                        yaml.dump(_t.tolist(), file)
                    self.cam_config.savePlanarize = False

                if self.cam_config.delPlanarize == True:
                    if os.path.exists("./aikensa/param/warptransform.yaml"):
                        os.remove("./aikensa/param/warptransform.yaml")
                    if os.path.exists("./aikensa/param/warptransform_lowres.yaml"):
                        os.remove("./aikensa/param/warptransform_lowres.yaml")
                    self.cam_config.delPlanarize = False

                if self.cam_config.saveImage == True:
                    os.makedirs("./aikensa/capturedimages", exist_ok=True)
                    os.makedirs("./aikensa/capturedimages/combinedImage", exist_ok=True)
                    os.makedirs("./aikensa/capturedimages/croppedFrame1", exist_ok=True)
                    os.makedirs("./aikensa/capturedimages/croppedFrame2", exist_ok=True)

                    combinedImage_dump = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"./aikensa/capturedimages/combinedImage/capturedimage_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", combinedImage_dump)

                    croppedFrame1_dump = cv2.cvtColor(croppedFrame1, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"./aikensa/capturedimages/croppedFrame1/croppedFrame1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", croppedFrame1_dump)

                    croppedFrame2_dump = cv2.cvtColor(croppedFrame2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"./aikensa/capturedimages/croppedFrame2/croppedFrame2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", croppedFrame2_dump)
                    print("Images saved.")
                    self.cam_config.saveImage = False


                clipFrame1 = self.frameCrop(frame1, x=590, y=0, w=600, h=600, wout = 128, hout = 128)
                clipFrame2 = self.frameCrop(frame1, x=1900, y=0, w=600, h=600, wout = 128, hout = 128)
                clipFrame3 = self.frameCrop(frame2, x=600, y=0, w=600, h=600, wout = 128, hout = 128)

                if self.cam_config.captureClip1:
                    clipFrame1_dump = cv2.cvtColor(clipFrame1, cv2.COLOR_BGR2RGB)
                    os.makedirs("./aikensa/capturedimages/clip1", exist_ok=True)
                    cv2.imwrite(f"./aikensa/capturedimages/clip1/clip1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", clipFrame1_dump)
                    self.cam_config.captureClip1 = False
                    print("Clip 1 captured.")
                if self.cam_config.captureClip2:
                    clipFrame2_dump = cv2.cvtColor(clipFrame2, cv2.COLOR_BGR2RGB)
                    os.makedirs("./aikensa/capturedimages/clip2", exist_ok=True)
                    cv2.imwrite(f"./aikensa/capturedimages/clip2/clip2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", clipFrame2_dump)
                    self.cam_config.captureClip2 = False
                    print("Clip 2 captured.")
                if self.cam_config.captureClip3:
                    clipFrame3_dump = cv2.cvtColor(clipFrame3, cv2.COLOR_BGR2RGB)
                    os.makedirs("./aikensa/capturedimages/clip3", exist_ok=True)
                    cv2.imwrite(f"./aikensa/capturedimages/clip3/clip3_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", clipFrame3_dump)
                    print("Clip 3 captured.")
                    self.cam_config.captureClip3 = False

                combinedImage_raw = combinedImage.copy()
                combinedImage = self.resizeImage(combinedImage, 1521, 363)
                
                croppedFrame1 = self.frameCrop(combinedImage_raw, x=450, y=260, w=320, h=160, wout = 320, hout = 160)
                croppedFrame2 = self.frameCrop(combinedImage_raw, x=3800, y=260, w=320, h=160, wout = 320, hout = 160)
                
                
                self.kata1Frame.emit(self.convertQImage(croppedFrame1))
                self.kata2Frame.emit(self.convertQImage(croppedFrame2))

                frame1_downres = self.resizeImage(frame1)
                frame2_downres = self.resizeImage(frame2)

                self.camFrame1.emit(self.convertQImage(frame1_downres))
                self.camFrame2.emit(self.convertQImage(frame2_downres))

                self.mergeFrame.emit(self.convertQImage(combinedImage))

                self.clip1Frame.emit(self.convertQImage(clipFrame1))
                self.clip2Frame.emit(self.convertQImage(clipFrame2))
                self.clip3Frame.emit(self.convertQImage(clipFrame3))

                

        cap_cam.release()
        print(f"Camera {self.calib_config.cameraID} released.")



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
  
    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def downSampling(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert resized cv2 image to QImage
        h, w, ch = resized_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(resized_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

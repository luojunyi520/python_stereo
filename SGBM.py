import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import stereoconfig


import cmath
import sys
import os
import cv2
import argparse
import random
import torch
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget

from stereo_SGBM import Ui_MainWindow
import cv2
import numpy as np
from PIL import Image

import colorsys
import os
import time

import numpy as np
import torch

import torch

torch.cuda.set_device(0)
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.efficientdet import EfficientDetBackbone
from utils.utils import (cvtColor, get_classes, image_sizes, preprocess_input,
                         resize_image)
from utils.utils_bbox import decodebox, non_max_suppression

class window(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.cwd = os.getcwd()
        self.setupUi(self)
        self.labels = [self.label_1, self.label_2, self.label_3]
        self.pushButton_pic1.clicked.connect(self.openimage)
        self.input_shape = [512, 512]
        self.class_names, self.num_classes = get_classes('model_data/voc_classes.txt')
        self.captureL = cv2.VideoCapture('left.mp4')
        self.captureR = cv2.VideoCapture('right.mp4')
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.letterbox_image=False
        self.points_3d = None
        self.pushButton_pic2.clicked.connect(self.SGBMpic)

        self.net = EfficientDetBackbone(self.num_classes, 0)
        self.net.load_state_dict(torch.load('logs/ep100-loss0.076-val_loss0.445.pth', map_location='cuda'))
        self.net = self.net.eval()
        self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()
        self.timer_video = QtCore.QTimer()  # 定时器
        self.timer_video.timeout.connect(self.show_video_frame)  # 每30毫秒调用一次视频显示

    def detect_image(self, image, crop=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)

            images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)

            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decodebox(regression, anchors, self.input_shape)
            results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=0.3,
                                          nms_thres=0.3)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            x = int((left+right)/2)
            y = int((top+bottom)/2)
            # print()
            d = self.points_3d[y][x]

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f} {} {:.2f}'.format(predicted_class, score,'dis', d[2])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)

            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decodebox(regression, anchors, self.input_shape)
            results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   传入网络当中进行预测
                # ---------------------------------------------------------#
                _, regression, classification, anchors = self.net(images)

                # -----------------------------------------------------------#
                #   将预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decodebox(regression, anchors, self.input_shape)
                results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   传入网络当中进行预测
            # ---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)

            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decodebox(regression, anchors, self.input_shape)
            results = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    def openimage(self):
        files, filetype = QFileDialog.getOpenFileNames(self, '打开多个图片', self.cwd,
                                                       "*.jpg, *.png, *.JPG, *.JPEG, All Files(*)")
        for i in range(len(files)):
            jpg = QtGui.QPixmap(files[i]).scaled(self.labels[i].width(), self.labels[i].height())
            print(jpg)
            self.labels[i].setPixmap(jpg)

    def show_video_frame(self):
        video_save_path = ""
        video_fps = 25.0
        fps = 0.0
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(self.captureL.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.captureL.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        refL, imgL = self.captureL.read()
        refR, imgR = self.captureR.read()
        imgLL,imgRR=imgL,imgR
        self.compute_disp(imgLL, imgRR)
        if refL and refR ==True:
            t1 = time.time()

            # frame=cv2.imread('1.png')

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(self.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
            self.labels[2].setPixmap(QPixmap.fromImage(self.Qframe))
        else:
            self.timer_video.stop()
            self.captureL.release()
            self.captureR.release()
    def compute_disp(self,imgLL,imgRR):
        imageSize = (1242, 375)
        cameraMatrixL = np.array([[721.5377, 0.0000, 609.5593],
                                  [0.0000, 721.5377, 172.8540],
                                  [0.0000, 0.0000, 1.0000]])
        distCoeffL = np.array([[0.0000, 0.0000, 0.0000,
                                0.0000, 0.0000]])
        cameraMatrixR = np.array([[721.5377, 0.0000, 609.5593],
                                  [0.0000, 721.5377, 172.8540],
                                  [0.0000, 0.0000, 1.0000]])
        distCoeffR = np.array([[0.0000, 0.0000, 0.0000,
                                0.0000, 0.0000]])
        # T平移向量
        T = np.array([[-540.0000], [0.0000], [0.0000]])

        # rec旋转向量
        R = np.array([[0.0000, 0.0098, -0.0074],
                      [-0.0098, 0.9999, -0.0043],
                      [0.0074, 0.0043, 9.999]])

        # 立体校正
        Rl, Rr, Pl, Pr, Q, validROIL, validROIR = cv2.stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR,
                                                                    distCoeffR,
                                                                    imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0,
                                                                    newImageSize=imageSize)

        start = time.clock()
        SGBM_blockSize = 5  # 一个匹配块的大小,大于1的奇数
        SGBM_num = 5
        min_disp = 0  # 最小的视差值，通常情况下为0
        num_disp = SGBM_num * 16  # 192 - min_disp #视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
        # blockSize = blockSize #匹配块大小（SADWindowSize），必须是大于等于1的奇数，一般为3~11
        uniquenessRatio = 0  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
        speckleRange = 0  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
        speckleWindowSize = 0  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
        disp12MaxDiff = 200  # 左右视差图的最大容许差异（超过将被清零），默认为 -1，即不执行左右视差检查。
        P1 = 600  # 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
        P2 = 2400  # p1控制视差平滑度，p2值越大，差异越平滑

        # grayImageL = cv2.cvtColor(imgLL, cv2.COLOR_BGR2GRAY)
        # grayImageR = cv2.cvtColor(imgRR, cv2.COLOR_BGR2GRAY)

        SGBM_stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,  # 最小的视差值
            numDisparities=num_disp,  # 视差范围
            blockSize=SGBM_blockSize,  # 匹配块大小（SADWindowSize）
            uniquenessRatio=uniquenessRatio,  # 视差唯一性百分比
            speckleRange=speckleRange,  # 视差变化阈值
            speckleWindowSize=speckleWindowSize,
            disp12MaxDiff=disp12MaxDiff,  # 左右视差图的最大容许差异
            P1=P1,  # 惩罚系数
            P2=P2
        )
        num_disp = SGBM_num * 16
        disp = SGBM_stereo.compute(imgLL, imgRR).astype(np.float32)
        disp = disp / 16.0
        self.points_3d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)

    def SGBMpic(self):
        self.timer_video.start(100)

    def getPixel(self,event):
        self.x = event.pos().x()
        self.y = event.pos().y()
        print(self.points_3d[self.y][self.x])
        d = self.points_3d[self.y][self.x]
        print('目标的深度距离为 %2f mm' % d[2])
        d_1 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        d_1 = d_1 ** 0.5
        print('目标距离拍摄点的直线距离为 %2f mm' % d_1)



    def preprocess(self,img1, img2):
        # 彩色图->灰度图
        if (img1.ndim == 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
        if (img2.ndim == 3):
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 直方图均衡
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)

        return img1, img2

    # 消除畸变
    def undistortion(self, image, camera_matrix, dist_coeff):

        undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

        return undistortion_image

    # 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
    # @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
    def getRectifyTransform(self,height, width, config):
        # 读取内参和外参
        left_K = config.cam_matrix_left
        right_K = config.cam_matrix_right
        left_distortion = config.distortion_l
        right_distortion = config.distortion_r
        R = config.R
        T = config.T

        # 计算校正变换
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                          (width, height), R, T, alpha=0)

        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

        return map1x, map1y, map2x, map2y, Q

    # 畸变校正和立体校正
    def rectifyImage(self,image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

        return rectifyed_img1, rectifyed_img2

    # 立体校正检验----画线
    def draw_line(self,image1, image2):
        # 建立输出图像
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]

        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:image1.shape[0], 0:image1.shape[1]] = image1
        output[0:image2.shape[0], image1.shape[1]:] = image2

        # 绘制等间距平行线
        line_interval = 50  # 直线间隔：50
        for k in range(height // line_interval):
            cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0),
                     thickness=2,
                     lineType=cv2.LINE_AA)

        return output

    # 视差计算
    def stereoMatchSGBM(self,left_image, right_image, down_scale=False):
        # SGBM匹配参数设置
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        blockSize = 3
        paraml = {'minDisparity': 0,
                  'numDisparities': 128,
                  'blockSize': blockSize,
                  'P1': 8 * img_channels * blockSize ** 2,
                  'P2': 32 * img_channels * blockSize ** 2,
                  'disp12MaxDiff': 1,
                  'preFilterCap': 63,
                  'uniquenessRatio': 15,
                  'speckleWindowSize': 100,
                  'speckleRange': 1,
                  'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }

        # 构建SGBM对象
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        paramr['minDisparity'] = -paraml['numDisparities']
        right_matcher = cv2.StereoSGBM_create(**paramr)

        # 计算视差图
        size = (left_image.shape[1], left_image.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
            disparity_right = right_matcher.compute(right_image, left_image)

        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]

            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
            disparity_right = factor * disparity_right

        # 真实视差（因为SGBM算法得到的视差是×16的）
        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.

        return trueDisp_left, trueDisp_right




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my = window()
    my.show()
    sys.exit(app.exec_())
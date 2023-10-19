#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from scipy import stats 


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        self.DepthFrameTrans = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3, dtype=float)
        # self.intrinsic_matrix = np.array([[898.831565, 0.000000, 709.284902], [0.000000, 897.087012, 402.250269], [0.000000, 0.000000, 1.000000]],dtype=float)
        self.extrinsic_matrix = np.eye(4, dtype=float)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = np.array([[-250.0, -25.0, 0.0], [-262.5, -37.5, 0.0],[-237.5, -37.5, 0.0],[-237.5, -12.5, 0.0],[-265.5, -12.5, 0.0],[250.0, -25.0, 0.0], [237.5, -37.5, 0.0],[262.5, -37.5, 0.0],[262.5, -12.5, 0.0],[237.5, -12.5, 0.0],[250.0, 275.0, 0.0],[237.5, 262.5, 0.0],[262.5, 262.5, 0.0],[262.5, 287.5, 0.0],[237.5, 287.5, 0.0], [-250.0, 275.0, 0.0],[-262.5, 262.5, 0.0],[-237.5, 262.5, 0.0],[-237.5, 287.5, 0.0],[-262.5, 287.5, 0.0],[-125.0,350.0,152.9],[-137.5,337.5,152.9],[-112.5,337.5,152.9],[-112.5,362.5,152.9],[-137.5,362.5,152.9],[125.0,-50.0,152.9],[112.5,-62.5,152.9],[137.5,-62.5,152.9],[137.5,-37.5,152.9],[112.5,-37.5,152.9]],dtype=float)
        self.Homography = np.eye(3)
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = {}
        self.w = None
        self.block_position_record = False
        self.detected_positions = []

        # rows, cols = 720, 1280

        # depth_offset_up_down = np.zeros((rows, cols), dtype=np.uint16)
        # depth_offset_right_left = np.zeros((rows, cols), dtype=np.uint16)
        # depth_range_up_down = np.linspace(38, -8, rows)
        # depth_range_right_left = np.linspace(-10, 10, cols)
        # # print(depth_range)
        # for i in range(rows):
        #     depth_offset_up_down[i, :] = depth_range_up_down[i]
        # for j in range(cols):
        #     depth_offset_right_left[:, j] = depth_range_right_left[j]
        
        # self.offset = depth_offset_up_down + depth_offset_right_left

        self.offset = np.zeros((720,1280)).astype(np.uint16)
        self.depthCalibrated = False

        # print(self.offset)

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        # self.DepthFrameHSV[..., 0] = np.right_shift(self.DepthFrameRaw.astype(np.uint16), 1)
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def uv2xyz_single(self, u, v):
        """!
        @brief      Convert u,v to x,y,z
                    return w 4x1 np.array

        """
        ori_pt = cv2.perspectiveTransform(np.array([[[u, v]]],dtype=np.float32), np.linalg.inv(self.Homography))
        d = np.array([[round(ori_pt[0][0][0])], [round(ori_pt[0][0][1])], [1.0]], dtype=float)
        z = self.DepthFrameRaw[round(ori_pt[0][0][1])][round(ori_pt[0][0][0])]
        H = np.array([[1.0,0.0,0.0,5.0],[0.0,-0.99,0.1225,240.0],[0.0,-0.1225,-0.99,1030.0],[0.0,0.0,0.0,1.0]], dtype=float)
        if self.cameraCalibrated:
            H = self.extrinsic_matrix
        k = np.array([[918.3599853515625, 0.0, 661.1923217773438], [0.0, 919.1538696289062, 356.59722900390625], [0.0, 0.0, 1.0]])
        c = np.matmul(np.linalg.inv(k*1/z),d)
        C = np.array([[c[0]], [c[1]], [c[2]],[1]], dtype=float)
        w = np.matmul(np.linalg.inv(H),C)
        return w
    


    def retrieve_area_color(self, hsv_data, contour, labels):
        mask = np.zeros(hsv_data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # mean_hsv = cv2.mean(hsv_data, mask=mask)[:3]
        # mean_hsv = stats.mode(hsv_data, mask=mask)[:3]
        
        # # Convert the mean hue value to an integer
        # mean_hue = int(mean_hsv[0])

        hue_channel = hsv_data[:, :, 0]  # Assuming hue is in the first channel (0)

        # Apply the mask to the hue channel
        hue_channel_masked = hue_channel[mask > 0]

        # Compute the mode of the hue values within the masked region
        mode_hue = int(stats.mode(hue_channel_masked).mode)

        median_hue = int(np.median(hue_channel_masked))
        
        min_dist = (np.inf, None)
        for label in labels:
            h_range = label["h_range"]
            
            h_within_range = h_range[0] <= median_hue <= h_range[1]
            
            if h_within_range:
                return label["id"], mode_hue
        
        return None, None


    def detectBlocksInDepthImage(self, image):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # colors = [
        #     {'id': 'red', 'h_range': (111, 180)},
        #     {'id': 'orange', 'h_range': (0, 24)},
        #     {'id': 'blue', 'h_range': (70, 91)},
        #     {'id': 'green', 'h_range': (33, 69)},
        #     {'id': 'yellow', 'h_range': (25, 32)},
        #     {'id': 'purple', 'h_range': (92, 110)}
        # ]

        colors = [
            {'id': 'red', 'h_range': (0, 2)},
            {'id': 'red', 'h_range': (162, 180)},
            {'id': 'orange', 'h_range': (3, 15)},
            {'id': 'blue', 'h_range': (96, 107)},
            {'id': 'green', 'h_range': (35, 80)},
            {'id': 'yellow', 'h_range': (16, 31)},
            {'id': 'purple', 'h_range': (108, 152)}
        ]

        # cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
        """mask out arm & outside board"""
        depth_data = self.DepthFrameTrans

        depth_data_with_offset = self.offset - depth_data

        mask = np.zeros_like(depth_data_with_offset, dtype=np.uint8)
        # mask = np.zeros_like(height_map_with_offset, dtype=np.uint8)
        cv2.rectangle(mask, (150,30),(1129,681), 255, cv2.FILLED)
        cv2.rectangle(mask, (540,385),(741,681), 0, cv2.FILLED)
        cv2.rectangle(image, (150,30),(1129,681), (255, 0, 0), 2)
        cv2.rectangle(image, (540,385),(741,681), (255, 0, 0), 2)
        
        lower = 15
        upper = 100

        thresh = cv2.bitwise_and(cv2.inRange(depth_data_with_offset, lower, upper), mask)
        # thresh = cv2.bitwise_and(cv2.inRange(height_map_with_offset, lower, upper), mask)

        # cv2.imshow("Threshold window", thresh)
        # # cv2.imshow("Image window", self.VideoFrame)
        # cv2.waitKey(0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply filtering to the contours
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Adjust these values
                filtered_contours.append(contour)

        # Define block detection criteria based on size or aspect ratio
        detected_blocks = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:
                detected_blocks.append(contour)

        # Draw only the detected blocks on the image
        cv2.drawContours(image, detected_blocks, -1, (0, 255, 0), thickness=2)
        hsv_image = cv2.cvtColor(self.VideoFrame.copy(), cv2.COLOR_BGR2HSV)
        if self.block_position_record:
            self.block_detections = {}

        for contour in detected_blocks:
            rgb_image = cv2.cvtColor(self.VideoFrame.copy(), cv2.COLOR_RGB2BGR)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            color, h = self.retrieve_area_color(hsv_image, contour, colors)
            area = cv2.contourArea(contour)
            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)
            cx = int(M['m10']/(M['m00']+1e-12))
            cy = int(M['m01']/(M['m00']+1e-12))
            cv2.putText(image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
            pos = self.uv2xyz_single(cx, cy)
            cv2.putText(image, str(int(pos[0])), (cx+30, cy-40), font, 1.0, (0,0,0), thickness=2)
            cv2.putText(image, str(int(pos[1])), (cx-30, cy-40), font, 1.0, (0,0,0), thickness=2)
            # pos = self.uv2xyz_single(cx, cy)
            # cv2.putText(image, str(int(pos[0])), (cx+30, cy-40), font, 1.0, (0,0,0), thickness=2)
            # cv2.putText(image, str(int(pos[1])), (cx+90, cy-40), font, 1.0, (0,0,0), thickness=2)
            # cv2.putText(image, str(int(cx)), (cx+30, cy-40), font, 1.0, (0,0,0), thickness=2)
            # cv2.putText(image, str(int(cy)), (cx+90, cy-40), font, 1.0, (0,0,0), thickness=2)

            # cv2.putText(image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
            # if h is not None:
            #     cv2.putText(image, str(int(h)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
            # print(color, int(theta), cx, cy)

            # Add blocks to the dictionary
            if self.block_position_record:
                pos = self.uv2xyz_single(cx, cy)

                self.detected_positions.append(pos[0:1])
                if self.block_detections.get(color) is not None:
                    self.block_detections[color].append([pos,theta,area])
                else:
                    self.block_detections[color] = [[pos,theta,area]]

        
        self.block_position_record = False
        # print(self.block_detections)
        # cv2.imshow("Threshold window", thresh)
        # cv2.imshow("Image window", hsv_image)
        # cv2.waitKey(0)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        if self.cameraCalibrated:
            # print("1", self.grid_points[0].shape)
            z_values = np.array(np.zeros_like(self.grid_points[0]))
            # print(z_values)
            grid_3d_points_homo = np.stack((self.grid_points[0], self.grid_points[1], z_values, np.ones_like(self.grid_points[0])))
            # print("2",grid_3d_points_homo.shape)
            grid_3d_points_homo = grid_3d_points_homo.reshape(4,-1)
            # print("3",grid_3d_points_homo.shape)
            pt_c = np.matmul(self.extrinsic_matrix, grid_3d_points_homo)
            # print("4",pt_c, np.shape(pt_c))
            proj = np.zeros((3,4), dtype=float)
            proj[:3,:3] = np.eye(3, dtype=float)
            pt_p = np.matmul(self.intrinsic_matrix, np.matmul(proj, pt_c))
            # print("5",pt_p.shape)
            for i in range(grid_3d_points_homo.shape[1]):
                pt_p[:4,i] = pt_p[:4,i]/pt_c[2,i]
            pixel = pt_p[:3,].T

            modified_image = self.VideoFrame.copy()
            for pt in pixel:
                center = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]],dtype=np.float32), self.Homography)
                cv2.circle(modified_image, [round(center[0][0][0]),round(center[0][0][1])], 5, (0, 255, 0), -1)
            self.GridFrame = modified_image
        else:
            self.GridFrame = self.VideoFrame.copy()
        # pass

     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # print(self.VideoFrame.shape)
        center_pts = np.array([])
        corner_pts = np.array([])
        n = len(msg.detections)

        if self.cameraCalibrated:
            for i in range(n):
                center = cv2.perspectiveTransform(np.array([[[msg.detections[i].centre.x, msg.detections[i].centre.y]]],dtype=np.float32), self.Homography)
                center_pts = np.append(center_pts, round(center[0][0][0]))
                center_pts = np.append(center_pts, round(center[0][0][1]))
                point1_tr = cv2.perspectiveTransform(np.array([[[msg.detections[i].corners[0].x, msg.detections[i].corners[0].y]]],dtype=np.float32), self.Homography)
                point1 = (round(point1_tr[0][0][0]), round(point1_tr[0][0][1]))
                point2_tr = cv2.perspectiveTransform(np.array([[[msg.detections[i].corners[1].x, msg.detections[i].corners[1].y]]],dtype=np.float32), self.Homography)
                point2 = (round(point2_tr[0][0][0]), round(point2_tr[0][0][1]))
                point3_tr = cv2.perspectiveTransform(np.array([[[msg.detections[i].corners[2].x, msg.detections[i].corners[2].y]]],dtype=np.float32), self.Homography)
                point3 = (round(point3_tr[0][0][0]), round(point3_tr[0][0][1]))
                point4_tr = cv2.perspectiveTransform(np.array([[[msg.detections[i].corners[3].x, msg.detections[i].corners[3].y]]],dtype=np.float32), self.Homography)
                point4 = (round(point4_tr[0][0][0]), round(point4_tr[0][0][1]))
                points = np.array([point1, point2, point3, point4], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(modified_image, [points], 5,(0, 0, 255), 2)
                cv2.putText(modified_image, 'ID: {}'.format(msg.detections[i].id), (round(point2_tr[0][0][0])+20, round(point2_tr[0][0][1])+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            center_pts = center_pts.reshape(n,2)
            for pt in center_pts:
                cv2.circle(modified_image, [round(pt[0]),round(pt[1])], 5, (0, 255, 0), -1)
        else:
            for i in range(n):
                center_pts = np.append(center_pts, round(msg.detections[i].centre.x))
                center_pts = np.append(center_pts, round(msg.detections[i].centre.y))
                point1 = (round(msg.detections[i].corners[0].x), round(msg.detections[i].corners[0].y))
                point2 = (round(msg.detections[i].corners[1].x), round(msg.detections[i].corners[1].y))
                point3 = (round(msg.detections[i].corners[2].x), round(msg.detections[i].corners[2].y))
                point4 = (round(msg.detections[i].corners[3].x), round(msg.detections[i].corners[3].y))
                points = np.array([point1, point2, point3, point4], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(modified_image, [points], 5,(0, 0, 255), 2)
                cv2.putText(modified_image, 'ID: {}'.format(msg.detections[i].id), (round(msg.detections[i].corners[1].x)+20, round(msg.detections[i].corners[1].y)+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            center_pts = center_pts.reshape(n,2)
            for pt in center_pts:
                cv2.circle(modified_image, [round(pt[0]),round(pt[1])], 5, (0, 255, 0), -1)

        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera
        self.counter = 0

    def callback(self, data):
        self.counter += 1
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        # self.camera.VideoFrame = cv_image
        if self.camera.cameraCalibrated:
            # image = self.camera.VideoFrame
            cv_image = cv2.warpPerspective(cv_image, self.camera.Homography, (cv_image.shape[1], cv_image.shape[0]))
            # if self.counter % 20 == 0:
            self.camera.detectBlocksInDepthImage(cv_image)
            self.camera.VideoFrame = cv_image
        else:
            self.camera.VideoFrame = cv_image
            # modified_image = self.camera.detectBlocksInDepthImage(cv_image)
            # self.camera.VideoFrame = modified_image

        

class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        # self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)
        pass


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth

        if self.camera.cameraCalibrated:
        
            #substract cv_depth by a slope offset of depth here to obtain cv_depth_with_offset
            # cv_depth_with_offset = cv_depth #+ self.camera.offset
            # print(cv_depth.dtype, cv_depth.shape)
            self.camera.DepthFrameTrans = cv2.warpPerspective(cv_depth, self.camera.Homography, (cv_depth.shape[1], cv_depth.shape[0]))
            # self.camera.DepthFrameRaw = cv2.warpPerspective(cv_depth_with_offset, self.camera.Homography, (cv_depth.shape[1], cv_depth.shape[0]))
        # else:
        #     self.camera.DepthFrameRaw = cv_depth

        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
"""
通过打呵欠进行疲劳检测
"""
import math
import os
import cv2
import dlib

# 我想检测这个人有没有打呵欠
class DetectWeary:
    def __init__(self):
        self.face_area = {
            "outline": [0, 16],
            "left_eyebrow": [17, 21],
            "right_eyebrow": [22, 26],
            "nose": [27, 35],
            "left_eye": [36, 41],
            "right_eye": [42, 47],
            "mouse": [48, 67]
        }
        self.EYERATIO = 0.25

    def zh_ch(self, string):
        return string.encode("gbk").decode(errors="ignore")

    def _showImg(self, img_name, img):
        """
        显示图片，并等待按键关闭窗口
        :param img_name:
        :param img:
        :return:
        """
        cv2.imshow(img_name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def detect_process(self,path):
        """
        检测图片流程
        :return:
        """
        # 1.读取图片并显示
        original_img = cv2.imread(path)
        # print("原始图像大小", original_img.shape)
        # self._showImg("original_img", original_img)
        # 2.灰度
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        # self._showImg("gray_img", gray_img)

        # 3.人脸检测
        # detector = dlib.get_frontal_face_detector()
        # # img = cv2.imread(path)
        # RGB_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # 由于opencv中读取的图片为BGR通道，需要转为RGB通道，再利用detector
        # 人脸分类器
        detector = dlib.get_frontal_face_detector()
        # 获取人脸检测器
        predictor = dlib.shape_predictor("68_face_landmarks.dat")

        faces = detector(original_img, 1)
        outline_pos, left_eyebrow, right_eyebrow, nose, left_eye, right_eye, mouse = [], [], [], [], [], [], []
        if not faces:
            img = cv2.putText(original_img, "No face has been detected!", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            self._showImg("img", img)
            return -1

        for face in faces:
            shape = predictor(original_img, face)  # 寻找人脸的68个标定点
            # 把脸换出来
            x1, x2 = (face.left(), face.top()), (face.right(), face.bottom())
            original_img = cv2.rectangle(original_img, x1, x2, (0,0,255), 2)
            for pt in shape.parts():
                x, y = pt.x, pt.y
                cv2.circle(original_img, (x, y), 2, (0,255,255), 1)
            # 遍历所有点，打印出其坐标，并圈出来
            outline_pos, left_eyebrow, right_eyebrow, nose, left_eye, right_eye, mouse = self.get_face_part(shape)
            # 计算眼睛长宽比
        left_eye_ratio = self._get_eye_ratio(left_eye)
        right_eye_ratio = self._get_eye_ratio(right_eye)
        mouse_ratio = self._get_mouse_ratio(mouse, original_img)
        # print(mouse_ratio)
        mouse, left_eye, right_eye = "No", "Open", "Open"
        if mouse_ratio > 0.6:
            mouse = 'Yes'
        if left_eye_ratio <= 0.25:
            left_eye = 'Closed'
        if right_eye_ratio <= 0.25:
            right_eye = "Colsed"
        s = '''yawn: %s %s  left eye: %s %s  right eye: %s %s''' % (mouse, round(mouse_ratio, 2),
                                                                    left_eye, round(left_eye_ratio, 2),
                                                                    right_eye, round(right_eye_ratio, 2))
        print(s)
        img = cv2.putText(original_img, s, (20, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.4, (255,255,255), 1)
        self._showImg("img", img)


    def _get_mouse_ratio(self, mouse, original_img):
        m1, m2, m3, m4, m5, m6, m7, m8 = mouse[-8:]
        width = self.get_euclidean_distance(m1, m5)  # 嘴巴横向距离
        heigh1 = self.get_euclidean_distance(m2, m8)  # 上下嘴唇距离
        heigh2 = self.get_euclidean_distance(m3, m7)  # 上下嘴唇距离
        heigh3 = self.get_euclidean_distance(m4, m6)  # 上下嘴唇距离
        avg_height = (heigh1 + heigh2 + heigh3) / 3
        mouse_ratio = avg_height / width
        return mouse_ratio


    def get_euclidean_distance(self, x1, x2):
        _x = x1[0] - x2[0]
        _y = x1[1] - x2[1]
        distance = math.sqrt(_x * _x + _y * _y)
        return distance

    def _get_eye_ratio(self, eye_list):
        """
        计算眼睛长宽比
        :return:
        """
        p1, p2, p3, p4, p5, p6 = eye_list
        dis_1_4 = self.get_euclidean_distance(p1, p4)
        dis_2_6 = self.get_euclidean_distance(p2, p6)
        dis_3_5 = self.get_euclidean_distance(p3, p5)
        eye_ratio = (dis_2_6 + dis_3_5) / (2*dis_1_4)
        return eye_ratio


    def get_face_part(self, shape):
        """
        获取五官位置点
        :param shape:
        :return:
        """
        keywords = shape.parts()
        outline_pos = [(pt.x, pt.y) for pt in keywords[:17]]
        left_eyebrow = [(pt.x, pt.y) for pt in keywords[17:22]]
        right_eyebrow = [(pt.x, pt.y) for pt in keywords[22:27]]
        nose = [(pt.x, pt.y) for pt in keywords[27:36]]
        left_eye = [(pt.x, pt.y) for pt in keywords[36:42]]
        right_eye = [(pt.x, pt.y) for pt in keywords[42:48]]
        mouse = [(pt.x, pt.y) for pt in keywords[48:]]
        return outline_pos, left_eyebrow,right_eyebrow,nose,left_eye,right_eye,mouse

detect_weary = DetectWeary()
path = "pic"
filelist = os.listdir(path)
pathlist = [os.path.join(path, file) for file in filelist if not file.startswith("__init__")]
# print(filelist)
for img_path in pathlist:
    detect_weary.detect_process(img_path)
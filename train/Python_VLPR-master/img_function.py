# -*- coding: utf-8 -*-
__author__ = '樱花落舞'
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import debug
import img_math
import img_recognition
import config

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = cv2.ml.SVM_load(fn)
        return self.model is not None

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.svm_path = self.base_dir / "svm.dat"
        self.svm_chinese_path = self.base_dir / "svmchinese.dat"
        self.chars2_dir = self.base_dir / "train" / "chars2"
        self.chars_chinese_dir = self.base_dir / "train" / "charsChinese"
        self.yolo_model = None
        self.yolo_model_path = None
        self.yolo_conf_thres = 0.35
        self.max_skew_correction_deg = 30.0
        self.min_skew_correction_deg = 0.5
        self.plate_target_size = (136, 36)
        self.reset_runtime_debug()

    def reset_runtime_debug(self):
        self.last_detection_boxes = []
        self.last_best_box = None
        self.last_pipeline_source = "none"
        self.last_debug_images = {}
        self.last_skew_angles = {"centroid": 0.0}

    def _set_debug_image(self, key, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return
        self.last_debug_images[key] = image.copy()

    def set_external_debug_image(self, key, image):
        self._set_debug_image(key, image)

    def get_runtime_debug(self):
        debug_images = {k: v.copy() for k, v in self.last_debug_images.items()}
        boxes = [dict(item) for item in self.last_detection_boxes]
        best_box = dict(self.last_best_box) if isinstance(self.last_best_box, dict) else None
        return {
            "boxes": boxes,
            "best_box": best_box,
            "pipeline_source": self.last_pipeline_source,
            "model_path": self.yolo_model_path,
            "skew_angles": dict(self.last_skew_angles),
            "debug_images": debug_images,
        }

    def __del__(self):
        try:
            self.save_traindata()
        except Exception:
            pass

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        loaded_ascii = False
        if self.svm_path.exists():
            try:
                loaded_ascii = bool(self.model.load(str(self.svm_path)))
            except Exception as e:
                print("svm.dat load failed, retrain will be used:", e)
                loaded_ascii = False

        if not loaded_ascii:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk(str(self.chars2_dir)):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    if digit_img is None:
                        continue
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            if len(chars_train) == 0:
                raise RuntimeError("未找到数字/字母训练样本，请检查目录: {}".format(self.chars2_dir))

            chars_train = list(map(img_recognition.deskew, chars_train))
            chars_train = img_recognition.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)

        loaded_chinese = False
        if self.svm_chinese_path.exists():
            try:
                loaded_chinese = bool(self.modelchinese.load(str(self.svm_chinese_path)))
            except Exception as e:
                print("svmchinese.dat load failed, retrain will be used:", e)
                loaded_chinese = False

        if not loaded_chinese:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk(str(self.chars_chinese_dir)):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = img_recognition.provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    if digit_img is None:
                        continue
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)

            if len(chars_train) == 0:
                raise RuntimeError("未找到中文训练样本，请检查目录: {}".format(self.chars_chinese_dir))

            chars_train = list(map(img_recognition.deskew, chars_train))
            chars_train = img_recognition.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if hasattr(self, "model") and self.model is not None and not self.svm_path.exists():
            self.model.save(str(self.svm_path))
        if hasattr(self, "modelchinese") and self.modelchinese is not None and not self.svm_chinese_path.exists():
            self.modelchinese.save(str(self.svm_chinese_path))

    def _find_default_yolo_model(self):
        ultra_root = Path(__file__).resolve().parents[1] / "ultralytics-SPD-SCSA"
        if not ultra_root.exists():
            return None

        candidates = sorted(ultra_root.glob("runs/detect/**/weights/best.pt"))
        if candidates:
            return candidates[-1]

        fallback = ultra_root / "yolo11n.pt"
        if fallback.exists():
            return fallback
        return None

    def _ensure_yolo_detector(self):
        if self.yolo_model is not None:
            return self.yolo_model

        model_path = self._find_default_yolo_model()
        if model_path is None:
            return None

        ultra_root = Path(__file__).resolve().parents[1] / "ultralytics-SPD-SCSA"
        if str(ultra_root) not in sys.path:
            sys.path.insert(0, str(ultra_root))

        try:
            from ultralytics import YOLO
        except Exception as e:
            print("ultralytics import failed:", e)
            return None

        try:
            self.yolo_model = YOLO(str(model_path))
            self.yolo_model_path = str(model_path)
            print("YOLO model loaded:", self.yolo_model_path)
        except Exception as e:
            print("YOLO model load failed:", e)
            self.yolo_model = None
        return self.yolo_model

    def _recognize_from_plate_roi(self, card_img, color):
        predict_result = []
        self.last_debug_images.pop("08_seg_binary", None)
        self.last_debug_images.pop("09_seg_peaks", None)
        if card_img is None or card_img.size == 0:
            return predict_result

        gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

        # 参考实现：黄绿车牌先反向，再做正向 OTSU 二值化。
        if color in ("green", "yello", "yellow"):
            gray_img = cv2.bitwise_not(gray_img)

        _, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._set_debug_image("08_seg_binary", cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR))
        # 先放一个默认波峰可视化，保证早退路径也能看到 09 图。
        self._set_debug_image("09_seg_peaks", cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR))

        x_histogram = np.sum(gray_img, axis=1)
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        x_threshold = (x_min + x_average) / 2

        wave_peaks = img_math.find_waves(x_threshold, x_histogram)
        if len(wave_peaks) == 0:
            return predict_result

        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        gray_img = gray_img[wave[0]:wave[1]]

        row_num, _ = gray_img.shape[:2]
        if row_num <= 2:
            return predict_result
        gray_img = gray_img[1:row_num - 1]

        y_histogram = np.sum(gray_img, axis=0)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram) / y_histogram.shape[0]
        y_threshold = (y_min + y_average) / 5
        wave_peaks = img_math.find_waves(y_threshold, y_histogram)

        seg_vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for start, end in wave_peaks:
            cv2.line(seg_vis, (start, 0), (start, seg_vis.shape[0] - 1), (0, 255, 255), 1)
            cv2.line(seg_vis, (end, 0), (end, seg_vis.shape[0] - 1), (0, 128, 255), 1)
        self._set_debug_image("09_seg_peaks", seg_vis)

        if len(wave_peaks) <= 6:
            return predict_result

        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        max_wave_dis = wave[1] - wave[0]
        if wave_peaks and wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
            wave_peaks.pop(0)

        cur_dis = 0
        i = 0
        for i, wave in enumerate(wave_peaks):
            if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                break
            else:
                cur_dis += wave[1] - wave[0]

        if i > 0:
            wave = (wave_peaks[0][0], wave_peaks[i][1])
            wave_peaks = wave_peaks[i + 1:]
            wave_peaks.insert(0, wave)

        if len(wave_peaks) < 3:
            return predict_result

        point = wave_peaks[2]
        point_img = gray_img[:, point[0]:point[1]]
        if np.mean(point_img) < 255 / 5:
            wave_peaks.pop(2)

        if len(wave_peaks) < 3:
            return predict_result

        part_cards = img_math.seperate_card(gray_img, wave_peaks)
        for i, part_card in enumerate(part_cards):
            if np.mean(part_card) < 255 / 5:
                continue
            part_card_old = part_card

            w = abs(part_card.shape[1] - SZ) // 2
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
            part_card = img_recognition.preprocess_hog([part_card])

            if i == 0:
                resp = self.modelchinese.predict(part_card)
                charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
            else:
                resp = self.model.predict(part_card)
                charactor = chr(int(resp[0]))

            if charactor == "1" and i == len(part_cards) - 1:
                if part_card_old.shape[0] / max(1, part_card_old.shape[1]) >= 7:
                    continue
            predict_result.append(charactor)

        return predict_result

    def _prepare_plate_binary_edges(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary, 50, 150)
        return binary, edges

    def _prepare_temp_angle_maps(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, temp_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(temp_binary, 50, 150)
        return gray, temp_binary, edges

    def _get_skew_angle_centroid_line(self, temp_binary):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp_binary, connectivity=8)
        pts = []
        h, w = temp_binary.shape[:2]
        min_area = max(8, int(h * w * 0.0008))
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area < min_area:
                continue
            if bh <= 0 or bw <= 0:
                continue
            ratio = bw / float(bh)
            # 过滤过细噪点和不太像字符块的连通域
            if ratio < 0.15 or ratio > 3.5:
                continue
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy)))

        if len(pts) < 2:
            return 0.0

        pts = np.array(pts, dtype=np.float32)
        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = float(np.degrees(np.arctan2(float(vy), float(vx))))

        if angle > 45:
            angle -= 90
        if angle < -45:
            angle += 90
        if abs(angle) > self.max_skew_correction_deg:
            return 0.0
        return angle

    def _deskew_plate(self, plate_img, angle):
        h, w = plate_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            plate_img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _estimate_skew_angle_traditional(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle = float(rect[-1])
        if angle < -45:
            angle += 90
        if angle > 45:
            angle -= 90
        if abs(angle) > self.max_skew_correction_deg:
            return 0.0
        return angle

    def _tight_crop_plate(self, plate_img):
        # 禁用过度裁剪，始终返回原 ROI。
        return plate_img

    def _normalize_plate_size(self, plate_img):
        target_w, target_h = self.plate_target_size
        if plate_img is None or plate_img.size == 0:
            return plate_img

        h, w = plate_img.shape[:2]
        if h <= 0 or w <= 0:
            return plate_img

        # 仅在尺寸超过目标时缩小，避免对小车牌强行放大导致字符发糊。
        scale = min(target_w / float(w), target_h / float(h))
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            return cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return plate_img

    def _mild_sharpen_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return plate_img
        # 使用低强度 unsharp，避免强核导致字符毛边和噪点被放大。
        blur = cv2.GaussianBlur(plate_img, (3, 3), 0)
        return cv2.addWeighted(plate_img, 1.12, blur, -0.12, 0)

    def _postprocess_yolo_plate_roi(self, roi):
        if roi is None or roi.size == 0:
            return roi

        working = roi.copy()
        angle = self._estimate_skew_angle_traditional(working)
        self.last_skew_angles = {"centroid": float(angle)}

        # 角度用于估计，但几何变换必须施加在彩色 ROI 上。
        if abs(angle) >= self.min_skew_correction_deg:
            working = self._deskew_plate(working, angle)
        else:
            working = working
        self._set_debug_image("04_plate_deskew", working)
        self._set_debug_image("04b_plate_deskew_refined", working)

        working = self._normalize_plate_size(working)
        self._set_debug_image("07_plate_normalized", working)
        working = self._mild_sharpen_plate(working)
        self._set_debug_image("07b_plate_sharpened", working)
        return working

    def _detect_plate_with_yolo(self, image_bgr):
        yolo_model = self._ensure_yolo_detector()
        if yolo_model is None:
            return None, None

        try:
            result = yolo_model(image_bgr, verbose=False)[0]
        except Exception as e:
            print("YOLO inference failed:", e)
            return None, None

        if result.boxes is None or len(result.boxes) == 0:
            return None, None

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
        confs = result.boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32) if result.boxes.cls is not None else None
        if confs.size == 0:
            return None, None

        names = result.names if hasattr(result, "names") else {}
        self.last_detection_boxes = []
        for idx, box in enumerate(boxes):
            x1_raw, y1_raw, x2_raw, y2_raw = map(int, box.tolist())
            bw = max(1, x2_raw - x1_raw)
            bh = max(1, y2_raw - y1_raw)
            conf = float(confs[idx])
            cls_id = int(cls_ids[idx]) if cls_ids is not None and idx < len(cls_ids) else -1
            if isinstance(names, dict):
                label = str(names.get(cls_id, cls_id))
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                label = str(names[cls_id])
            else:
                label = "plate"
            self.last_detection_boxes.append({
                "x": x1_raw,
                "y": y1_raw,
                "w": bw,
                "h": bh,
                "conf": conf,
                "label": label,
            })

        self.last_detection_boxes.sort(key=lambda item: item["conf"], reverse=True)

        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        if best_conf < self.yolo_conf_thres:
            return None, None

        x1, y1, x2, y2 = map(int, boxes[best_idx].tolist())
        h, w = image_bgr.shape[:2]
        # 轻微外扩检测框，给矫正与分割保留安全边界。
        expand = 0
        x1 = max(0, min(x1 - expand, w - 1))
        y1 = max(0, min(y1 - expand, h - 1))
        x2 = min(w, max(x2 + expand, x1 + 1))
        y2 = min(h, max(y2 + expand, y1 + 1))
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)

        self.last_best_box = {
            "x": x1,
            "y": y1,
            "w": max(1, x2 - x1),
            "h": max(1, y2 - y1),
            "conf": best_conf,
            "label": self.last_detection_boxes[0]["label"] if len(self.last_detection_boxes) else "plate",
        }

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        self._set_debug_image("01_yolo_raw_crop", roi)

        # YOLO 只负责粗定位，后续细化定位由传统方法在该 ROI 内完成。
        return roi, "no"

    def img_first_pre(self, car_pic_file):
        """
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # 缩小图片

        blur = 3
        img = cv2.GaussianBlur(img, (blur, blur), 0)
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转化成灰度图像

        Matrix = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        # 创建20*20的元素为1的矩阵 开操作，并和img重合

        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_edge = cv2.Canny(img_thresh, 100, 200)
        # Otsu’s二值化 找到图像边缘

        Matrix = np.ones((4, 19), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        return img_edge2, oldimg

    def img_color_contours(self, img_contours, oldimg, rawimg=None):
        """
        :param img_contours: 预处理好的图像
        :param oldimg: 原图像
        :return: 已经定位好的车牌
        """

        # 使用 YOLO 粗定位，再在 YOLO ROI 内执行传统检测细化车牌区域。
        yolo_input = rawimg if rawimg is not None else oldimg
        yolo_roi, _ = self._detect_plate_with_yolo(yolo_input)
        if yolo_roi is not None:
            roi_edges, roi_old = self.img_first_pre(yolo_roi)
            if roi_edges.any():
                config.set_name(roi_edges)

            pic_hight, pic_width = roi_edges.shape[:2]
            card_contours = img_math.img_findContours(roi_edges)
            card_imgs = img_math.img_Transform(card_contours, roi_old, pic_width, pic_hight)
            colors, _ = img_math.img_color(card_imgs)

            for i, color in enumerate(colors):
                if color in ("blue", "yello", "green"):
                    refined_card = self._postprocess_yolo_plate_roi(card_imgs[i])
                    yolo_result = self._recognize_from_plate_roi(refined_card, color)
                    if len(yolo_result) > 0:
                        self.last_pipeline_source = "yolo_traditional"
                        return yolo_result, refined_card, color

            # 传统细化失败时，回退使用 YOLO ROI 直接识别，避免整张图无结果。
            fallback_roi = self._postprocess_yolo_plate_roi(yolo_roi)
            fallback_colors, _ = img_math.img_color([fallback_roi.copy()])
            fallback_color = fallback_colors[0] if len(fallback_colors) else "blue"
            if fallback_color in ("blue", "yello", "green"):
                fallback_result = self._recognize_from_plate_roi(fallback_roi, fallback_color)
                if len(fallback_result) > 0:
                    self.last_pipeline_source = "yolo_fallback"
                    return fallback_result, fallback_roi, fallback_color

        self.last_pipeline_source = "yolo_traditional_no_result"
        return [], None, None  # 识别到的字符、定位的车牌图像、车牌颜色

    def img_only_color(self, filename, oldimg, img_contours):
        """
        :param filename: 图像文件
        :param oldimg: 原图像文件
        :return: 已经定位好的车牌
        """
        pic_hight, pic_width = img_contours.shape[:2]
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # 根据阈值找到对应颜色

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)

        card_contours = img_math.img_findContours(img_edge2)
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        colors, car_imgs = img_math.img_color(card_imgs)

        predict_result = []
        roi = None
        card_color = None

        for i, color in enumerate(colors):
            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                # 颜色分支复用统一分割识别逻辑，避免第三套分割代码导致结果漂移。
                predict_result = self._recognize_from_plate_roi(card_img, color)
                roi = card_img
                card_color = color
                self.last_pipeline_source = "traditional_color"
                break
        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色

    def img_mser(self, filename):
        if type(filename) == type(""):
            img = img_math.img_read(filename)
        else:
            img = filename
        oldimg = img
        mser = cv2.MSER_create(_min_area=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions, boxes = mser.detectRegions(gray)
        colors_img = []
        for box in boxes:
            x, y, w, h = box
            width, height = w, h
            if width < height:
                width, height = height, width
            ration = width / height

            if w * h > 1500 and 3 < ration < 4 and w > h:
                cropimg = img[y:y + h, x:x + w]
                colors_img.append(cropimg)

        debug.img_show(img)
        colors, car_imgs = img_math.img_color(colors_img)
        for i, color in enumerate(colors):
            if color != "no":
                print(color)
                debug.img_show(car_imgs[i])

# -*- coding: utf-8 -*-
__author__ = '樱花落舞'

import threading
import time
import traceback
import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import img_function as predict


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

predictor_lock = threading.Lock()
predictor = predict.CardPredictor()
predictor.train_svm()

PROCESS_IMAGE_TITLES = {
    "01_yolo_raw_crop": "YOLO原始裁剪",
    "03b_angle_binary": "角度估计二值图",
    "04_plate_deskew": "车牌倾斜矫正",
    "04b_border_filled": "旋转后黑边填充",
    "07_plate_normalized": "车牌尺寸归一化",
    "07b_plate_sharpened": "车牌轻微锐化",
    "08_seg_binary": "字符分割二值图",
    "09_seg_peaks": "字符分割波峰图",
}


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def _encode_image_b64(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + payload


def _recognize_text(img_bgr):
    predictor.reset_runtime_debug()

    try:
        # 纯 YOLO 定位链路：直接使用原图做检测与 ROI 裁剪，不再依赖传统预处理输出。
        r_shape, _, _ = predictor.img_color_contours(img_bgr, img_bgr, rawimg=img_bgr)
    except Exception:
        traceback.print_exc()
        r_shape = []

    text = _as_text(r_shape)

    runtime_debug = predictor.get_runtime_debug()
    return text, runtime_debug


@app.get("/")
def index_page():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.post("/api/plate/recognize")
def recognize_api():
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"message": "缺少上传文件"}), 400

    raw = upload.read()
    if not raw:
        return jsonify({"message": "上传文件为空"}), 400

    img_array = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"message": "无法解析图片"}), 400

    start = time.time()
    try:
        with predictor_lock:
            text, runtime_debug = _recognize_text(img_bgr)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": "识别失败", "detail": str(e)}), 500

    process_images = []
    seen_keys = set()
    debug_images = runtime_debug.get("debug_images", {}) if isinstance(runtime_debug, dict) else {}
    for key in sorted(debug_images.keys()):
        if key in seen_keys:
            continue
        src = _encode_image_b64(debug_images.get(key))
        if not src:
            continue
        seen_keys.add(key)
        process_images.append({
            "key": key,
            "title": PROCESS_IMAGE_TITLES.get(key, key),
            "src": src,
        })

    boxes = runtime_debug.get("boxes", []) if isinstance(runtime_debug, dict) else []
    best_box = runtime_debug.get("best_box") if isinstance(runtime_debug, dict) else None
    detector = {
        "source": runtime_debug.get("pipeline_source", "none") if isinstance(runtime_debug, dict) else "none",
        "modelPath": runtime_debug.get("model_path") if isinstance(runtime_debug, dict) else None,
        "confidence": best_box.get("conf") if isinstance(best_box, dict) else None,
        "skewAngles": runtime_debug.get("skew_angles", {}) if isinstance(runtime_debug, dict) else {},
    }

    elapsed_ms = int((time.time() - start) * 1000)
    return jsonify({
        "text": text,
        "elapsedMs": elapsed_ms,
        "boxes": boxes,
        "bestBox": best_box,
        "detector": detector,
        "processImages": process_images,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import time
import cv2
import numpy as np

app = FastAPI()

try:
    import easyocr  # lightweight OCR that bundles detection
    import torch
    _use_gpu = torch.cuda.is_available()
    if _use_gpu:
        print("[info] GPU available, using CUDA for OCR")
    _reader = easyocr.Reader(['ch_sim', 'en'], gpu=_use_gpu)
    OCR_READY = True
except Exception as exc:  # pragma: no cover
    print("[warn] easyocr not available: %s" % exc)
    OCR_READY = False
    _reader = None


def find_plate_box(image_bgr):
    """Heuristic plate detection: edge -> contour filter by aspect ratio."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 500:  # skip tiny
            continue
        aspect = w / float(h)
        if 1.8 <= aspect <= 6.0:  # typical plate ratio
            candidates.append((w * h, (x, y, w, h)))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)

    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    t0 = time.time()

    box = find_plate_box(img)
    cropped = None
    if box:
        x, y, w, h = box
        cropped = img[y:y + h, x:x + w]
    else:
        # fallback to whole image
        x, y, w, h = 0, 0, img.shape[1], img.shape[0]
        cropped = img

    text = ""
    if OCR_READY:
        try:
            ocr_res = _reader.readtext(cropped)
            if ocr_res:
                # take the highest score line
                ocr_res.sort(key=lambda r: r[2], reverse=True)
                text = ocr_res[0][1]
        except Exception as exc:  # pragma: no cover
            print("[warn] OCR failed: %s" % exc)

    elapsed = int((time.time() - t0) * 1000)
    result = {
        "text": text or "未识别",
        "boxes": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 0.5}] if box else [],
        "elapsedMs": elapsed,
    }

    if not OCR_READY:
        result["note"] = "easyocr 未安装，返回占位结果"

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

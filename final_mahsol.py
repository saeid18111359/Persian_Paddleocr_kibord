import io
import os
import sys 
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
import re 
import warnings
import cv2
import numpy as np

os.environ["FLAGS_use_mkldnn"] = "0" #جلوگیری از اختلالات
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#جلوگیری از خطا به دلیل وجود چند نسخه
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"#غیرفعال سازی جستجو انلاین

warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, HTTPException, UploadFile,BackgroundTasks
from fastapi.responses import JSONResponse
from hazm import Normalizer
from paddleocr import PaddleOCR
import uvicorn

app = FastAPI(
    title="OCR End-to-End Pipeline - Keyboard AI",
    version="1.5.5",
    contact={
        "name": "Rozhan Agha Alizadeh",   
    }
)

DET_MODEL_DIR = r"C:\Users\CodePad\.paddleocr\whl\det\ch\ch_PP-OCRv4_det_infer"
REC_MODEL_DIR = r"C:\Users\CodePad\Desktop\ocr\faze2_test3\PaddleOCR\inference_model_rec_final6"
DICT_PATH = r"C:\Users\CodePad\Desktop\ocr\faze2_test3\rec_photos\dict.txt"

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    det_model_dir=DET_MODEL_DIR,
    rec_model_dir=REC_MODEL_DIR,
    rec_char_dict_path=DICT_PATH,
    det_limit_side_len=1280,#حداکثر اندازه ضلع تصویر برای det
    det_db_thresh=0.25,    #آستانه تشخیص پیکسل‌های متن
    det_db_box_thresh=0.45,#حداقل امتیاز لازم برای قبول یک باکس متن
    det_db_unclip_ratio=2.4,#   میزان گسترش کادر متن بعد از تشخیص
    det_db_score_mode='slow',   
    device="cpu"
)

hazm_normalizer = Normalizer()

def clean_ctc_stutter(text: str) -> str:#جلوگیری از تکرار واژه
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
    text = re.sub(r"(\w{2,})\1$", r"\1", text)
    return text


def process_and_normalize_text(text: str) -> str:
    cleaned = clean_ctc_stutter(text)
    normalized = hazm_normalizer.normalize(cleaned)
    return normalized.strip()


def group_and_merge_boxes(boxes_with_res):

    if not boxes_with_res:
        return []

    parsed_items = []
    heights = []
    
    for res in boxes_with_res:
        box, (text, score) = res
        pts = np.array(box, dtype=np.float32)
        max_x = np.max(pts[:, 0])#سمت راست ترین گوشه کادر
        min_y = np.min(pts[:, 1])#گوشه های پایین کادر
        max_y = np.max(pts[:, 1])# گوشه های بالا
        center_y = (min_y + max_y) / 2.0#مرکز عمودی کادر
        box_height = max_y - min_y
        
        heights.append(box_height)
        parsed_items.append({
            "res": res, 
            "max_x": max_x, 
            "center_y": center_y, 
            "box": box
        })

    avg_h = float(np.mean(heights)) if heights else 20.0# بررسی ارتفاع میانگین باکس ها
    line_threshold = avg_h * 0.5  # اگر اختلاف مرکز کادر ها کمتر از نصف میانگین باشد یعنی در یک سطر هستند
    x_threshold = avg_h * 0.45 # همان کار بالا را انجام میدهد ولی در سطر   

    parsed_items.sort(key=lambda b: b["center_y"])
    lines = []
   
    for item in parsed_items:
        placed = False
        for line in lines:
            if abs(line["avg_y"] - item["center_y"]) < line_threshold:
                line["items"].append(item)
                line["avg_y"] = sum(i["center_y"] for i in line["items"]) / len(line["items"])
                placed = True
                break
        if not placed:
            lines.append({"avg_y": item["center_y"], "items": [item]})

    sorted_results = []
    for line in lines:
        line["items"].sort(key=lambda item: item["max_x"], reverse=True)#مرتب سازی از راست ترین به چپ ترین
        
        if not line["items"]:
            continue
            
        merged_line = []
        current_item = line["items"][0]["res"] # راست ترین کلمه در سطر اول
        
        for next_item in line["items"][1:]:
            next_res = next_item["res"]
            curr_box, (curr_text, curr_score) = current_item
            next_box, (next_text, next_score) = next_res
            # box = [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]
            curr_left = min(p[0] for p in curr_box)
            next_right = max(p[0] for p in next_box)
            distance = curr_left - next_right
            
            if 0 <= distance <= x_threshold or distance < 0:
                '''all_points = [
                     [x1, y1], [x2, y2], [x3, y3], [x4, y4],
                     [x5, y5], [x6, y6], [x7, y7], [x8, y8]]
                '''
                all_points = curr_box + next_box
                new_min_x = float(min(p[0] for p in all_points))
                new_max_x = float(max(p[0] for p in all_points))
                new_min_y = float(min(p[1] for p in all_points))
                new_max_y = float(max(p[1] for p in all_points))
                
                merged_box = [
                    [new_min_x, new_min_y],
                    [new_max_x, new_min_y],
                    [new_max_x, new_max_y],
                    [new_min_x, new_max_y]
                ]
                merged_text = curr_text + " " + next_text
                merged_score = (curr_score + next_score) / 2.0
                
                current_item = [merged_box, (merged_text, merged_score)]
            else:
                merged_line.append(current_item)
                current_item = next_res
                
        merged_line.append(current_item)
        sorted_results.extend(merged_line)

    return sorted_results
'''[100, 30], [50, 40], [100, 40], [50, 30],
                     [200, 100], [200, 70], [150, 100], [150, 70]]
                     
                     x_min = 150
                     x_max = 100
                     ...
                     area = 0 
'''
def calculate_iou(box1, box2):
    p1 = np.array(box1, dtype=np.float32)
    p2 = np.array(box2, dtype=np.float32)

    x1_min, y1_min = np.min(p1, axis=0)
    x1_max, y1_max = np.max(p1, axis=0)

    x2_min, y2_min = np.min(p2, axis=0)
    x2_max, y2_max = np.max(p2, axis=0)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def box_process(result):
    if not result or not result[0]:
        return []

    sorted_results = group_and_merge_boxes(result[0])

    final_outputs = []
    for res in sorted_results:
        box, (raw_text, score) = res

        if score < 0.60:
            continue

        clean_text = process_and_normalize_text(raw_text)

        if not clean_text or (len(clean_text) <= 1 and not clean_text.isdigit()):
            continue

        final_outputs.append({
            "text": clean_text,
            "confidence": float(score),
            "box": box
        })
    return final_outputs


def deskew_image(img_cv: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img_cv
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    if abs(angle) > 0.3:
        (h, w) = img_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_cv = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_cv

def dynamic_resize(img_cv: np.ndarray) -> np.ndarray:
    h, w = img_cv.shape[:2]
    max_dim = 1600
    min_dim = 600
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif max(h, w) < min_dim:
        scale = min_dim / min(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img_cv

def advanced_binarization(gray_img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    blended = cv2.addWeighted(enhanced, 0.6, adaptive_thresh, 0.4, 0)
    return blended
def preprocess_image(img_cv: np.ndarray) -> np.ndarray:
    if img_cv is None:
        return None

    resized = dynamic_resize(img_cv)
    deskewed = deskew_image(resized)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    binarized = advanced_binarization(denoised)
    
    gaussian_3 = cv2.GaussianBlur(binarized, (0, 0), 2.0)
    sharpened = cv2.addWeighted(binarized, 1.5, gaussian_3, -0.5, 0)
    
    processed_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return processed_bgr


from fastapi.responses import HTMLResponse 
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>سیستم هوشمند OCR | کیبورد هوش مصنوعی</title>
        <link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet" type="text/css" />
        <style>
            :root {
                --bg-color: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-muted: #64748b;
                --primary: #4f46e5;
                --primary-hover: #4338ca;
                --border: #e2e8f0;
                --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Vazirmatn', sans-serif;
            }

            body {
                background-color: var(--bg-color);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem 1rem;
            }

            .container {
                width: 100%;
                max-width: 800px;
            }

            header {
                text-align: center;
                margin-bottom: 2rem;
            }

            header h1 {
                font-size: 1.8rem;
                color: var(--primary);
                margin-bottom: 0.5rem;
            }

            header p {
                color: var(--text-muted);
                font-size: 0.95rem;
            }

            .card {
                background: var(--card-bg);
                border-radius: 1rem;
                box-shadow: var(--shadow);
                padding: 2rem;
                border: 1px solid var(--border);
                margin-bottom: 1.5rem;
            }

            .upload-area {
                border: 2px dashed var(--border);
                border-radius: 0.75rem;
                padding: 2.5rem 1rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #fafafa;
            }

            .upload-area:hover, .upload-area.dragover {
                border-color: var(--primary);
                background: #eef2ff;
            }

            .upload-area input {
                display: none;
            }

            .upload-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
            }

            .upload-text {
                font-weight: 500;
                margin-bottom: 0.5rem;
            }

            .upload-hint {
                font-size: 0.85rem;
                color: var(--text-muted);
            }

            .preview-container {
                margin-top: 1.5rem;
                text-align: center;
                display: none;
            }

            .preview-image {
                max-width: 100%;
                max-height: 250px;
                border-radius: 0.5rem;
                border: 1px solid var(--border);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            }

            .btn {
                display: block;
                width: 100%;
                padding: 0.85rem;
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 0.75rem;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.2s;
                margin-top: 1.5rem;
                text-align: center;
            }

            .btn:hover {
                background-color: var(--primary-hover);
            }

            .btn:disabled {
                background-color: var(--text-muted);
                cursor: not-allowed;
            }

            .loading {
                display: none;
                text-align: center;
                margin: 1.5rem 0;
                color: var(--primary);
                font-weight: 500;
            }

            .spinner {
                display: inline-block;
                width: 2rem;
                height: 2rem;
                border: 3px solid rgba(79, 70, 229, 0.2);
                border-radius: 50%;
                border-top-color: var(--primary);
                animation: spin 0.8s linear infinite;
                margin-bottom: 0.5rem;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .results-section {
                display: none;
            }

            .results-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .results-header h3 {
                font-size: 1.1rem;
            }

            .copy-btn {
                background: #e2e8f0;
                color: var(--text-main);
                border: none;
                padding: 0.4rem 0.8rem;
                border-radius: 0.5rem;
                font-size: 0.85rem;
                cursor: pointer;
                transition: background 0.2s;
            }

            .copy-btn:hover {
                background: #cbd5e1;
            }

            .results-box {
                background: #f8fafc;
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                padding: 1rem;
                max-height: 300px;
                overflow-y: auto;
                font-size: 0.95rem;
                line-height: 1.8;
                white-space: pre-wrap;
                word-break: break-word;
            }

            footer {
                text-align: center;
                margin-top: auto;
                color: var(--text-muted);
                font-size: 0.85rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>سیستم هوشمند استخراج متن (OCR)</h1>
                <p>موتور  پردازش تصاویر محیطی - تیم 36 روژان آقاعلیزاده</p>
            </header>

            <div class="card">
                <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📄</div>
                    <div class="upload-text">تصویر خود را اینجا رها کنید یا کلیک کنید</div>
                    <div class="upload-hint">فرمت‌های پشتیبانی شده: JPG, PNG, WEBP</div>
                    <input type="file" id="fileInput" accept="image/*">
                </div>

                <div class="preview-container" id="previewContainer">
                    <img id="imagePreview" class="preview-image" alt="پیش‌نمایش تصویر">
                </div>

                <button id="submitBtn" class="btn" style="display: none;" onclick="uploadImage()">شروع پردازش و استخراج متن</button>
                <button id="shutdownBtn" class="btn" style="background-color: #ef4444; margin-top: 0.75rem;" onclick="shutdownApp()">بستن و خروج کامل از برنامه</button>
            </div>

            <div class="loading" id="loadingIndicator">
                <div class="spinner"></div>
                <div>در حال تحلیل و استخراج متن تصویر، لطفاً صبر کنید...</div>
            </div>

            <div class="card results-section" id="resultsSection">
                <div class="results-header">
                    <h3>متن استخراج‌شده:</h3>
                    <button class="copy-btn" onclick="copyResults()">کپی متن</button>
                </div>
                <div class="results-box" id="resultsText"></div>
            </div>
        </div>

        <footer>
            طراحی شده با ❤️   | نسخه 1.5.5
        </footer>

        <script>
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('fileInput');
            const previewContainer = document.getElementById('previewContainer');
            const imagePreview = document.getElementById('imagePreview');
            const submitBtn = document.getElementById('submitBtn');
            const loadingIndicator = document.getElementById('loadingIndicator');
            const resultsSection = document.getElementById('resultsSection');
            const resultsText = document.getElementById('resultsText');

            let selectedFile = null;

            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropArea.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropArea.classList.remove('dragover');
                }, false);
            });

            dropArea.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length) {
                    handleFile(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) {
                    handleFile(e.target.files[0]);
                }
            });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert('لطفاً یک فایل تصویر معتبر انتخاب کنید.');
                    return;
                }
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    previewContainer.style.display = 'block';
                    submitBtn.style.display = 'block';
                    resultsSection.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }

            async function shutdownApp() {
                if (confirm('آیا مطمئنید می‌خواهید برنامه را ببندید؟ پورت آزاد خواهد شد.')) {
                    try {
                        await fetch('/shutdown', { method: 'POST' });
                        alert('برنامه با موفقیت بسته شد. حالا می‌توانید این صفحه را ببندید.');
                        window.close();
                    } catch (e) {
                        alert('برنامه بسته شد.');
                    }
                }
            }

            async function uploadImage() {
                if (!selectedFile) return;

                const formData = new FormData();
                formData.append('file', selectedFile);

                loadingIndicator.style.display = 'block';
                submitBtn.style.display = 'none';
                resultsSection.style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    loadingIndicator.style.display = 'none';
                    submitBtn.style.display = 'block';

                    if (response.ok && data.status === 'success') {
                        const fullText = data.results.map(item => item.text).join('\\n');
                        resultsText.textContent = fullText || 'هیچ متنی در تصویر تشخیص داده نشد.';
                        resultsSection.style.display = 'block';
                    } else {
                        alert(data.message || 'خطایی در پردازش تصویر رخ داد.');
                    }
                } catch (error) {
                    loadingIndicator.style.display = 'none';
                    submitBtn.style.display = 'block';
                    alert('خطا در ارتباط با سرور: ' + error.message);
                }
            }

            function copyResults() {
                const text = resultsText.textContent;
                navigator.clipboard.writeText(text).then(() => {
                    const btn = document.querySelector('.copy-btn');
                    btn.textContent = 'کپی شد!';
                    setTimeout(() => {
                        btn.textContent = 'کپی متن';
                    }, 2000);
                });
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict_ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="فایل ارسالی باید تصویر باشد.")

    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            return JSONResponse(content={"status": "error", "message": "فایل تصویری معتبر نیست"}, status_code=400)
        res1 = ocr_engine.ocr(img_cv)
        final_outputs = box_process(res1)
        
        return JSONResponse(
            content={
                "status": "success",
                "total_detected": len(final_outputs),
                "results": final_outputs,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش تصویر: {str(e)}")
@app.post("/shutdown")
def shutdown_server(background_tasks: BackgroundTasks):
    def kill_process():
        os._exit(0) 
    
    background_tasks.add_task(kill_process)
    return {"status": "success", "message": "برنامه با موفقیت بسته شد."}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090)
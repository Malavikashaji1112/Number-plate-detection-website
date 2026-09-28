import asyncio
import threading
import queue as queue_module
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import os
import re
import uuid
import base64
import torch

app = FastAPI(title="Number Plate Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

print("Loading vehicle model...")
vehicle_model = YOLO("yolo26n.pt")
vehicle_model.to(DEVICE)

print("Loading number plate model...")
plate_model = YOLO("license_plate_detector.pt")
plate_model.to(DEVICE)

print("Loading PaddleOCR...")
reader = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)
print("Models loaded successfully.")

vehicle_classes = [2, 3, 5, 7]
MIN_CONF = 0.6
DETECT_PLATE_EVERY_N_FRAMES = 10
VEHICLE_DETECT_EVERY_N_FRAMES = 2
MAX_FRAME_WIDTH = 960
VEHICLE_IMGSZ = 640
FRAME_JPEG_QUALITY = 60
SEND_FRAME_EVERY_N_PROCESSED = 1  

PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')


def process_video_streaming(video_path):
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open uploaded video.")

    plates = {}
    plate_boxes = {}
    boxes = {}
    frame_idx = 0
    processed_idx = 0

    while cap.isOpened():
        success, frame = cap.read() #frame is a numpy array holding the pixel data,shape(h,w,3)
        if not success:
            break

        frame_idx += 1

        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / w
            frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

        if frame_idx % VEHICLE_DETECT_EVERY_N_FRAMES != 0:
            continue

        processed_idx += 1

        vehicle_results = vehicle_model.track(
            frame, classes=vehicle_classes, persist=True, conf=MIN_CONF,
            imgsz=VEHICLE_IMGSZ, device=DEVICE, verbose=False
        )

        boxes = {}
        if vehicle_results[0].boxes.id is not None:
            for vbox, car_id in zip(vehicle_results[0].boxes, vehicle_results[0].boxes.id.int().tolist()):
                vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
                vx1 = max(0, vx1)
                vy1 = max(0, vy1)
                vx2 = min(frame.shape[1], vx2)
                vy2 = min(frame.shape[0], vy2)
                boxes[car_id] = (vx1, vy1, vx2, vy2)

                if car_id in plates:
                    continue

                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0:
                    continue

                need_box_refresh = (
                    car_id not in plate_boxes or frame_idx % DETECT_PLATE_EVERY_N_FRAMES == 0
                )

                if need_box_refresh:
                    plate_results = plate_model(vehicle_crop, conf=0.3, device=DEVICE, verbose=False)

                    for presults in plate_results:
                        for pbox in presults.boxes:
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                            px1 = max(0, px1)
                            py1 = max(0, py1)
                            px2 = min(vehicle_crop.shape[1], px2)
                            py2 = min(vehicle_crop.shape[0], py2)

                            plate_crop = vehicle_crop[py1:py2, px1:px2]
                            if plate_crop.size == 0:
                                continue

                            plate_boxes[car_id] = (px1, py1, px2, py2)

                            plate_crop_up = cv2.resize(
                                plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                            )
                            ocr_result = reader.predict(plate_crop_up)

                            try:
                                texts = ocr_result[0]["rec_texts"]
                                scores = ocr_result[0]["rec_scores"]
                            except Exception:
                                texts, scores = [], []

                            plate_text = "".join(texts).replace(" ", "").upper()
                            plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)

                            already_seen = {info["plate"] for info in plates.values()}
                            if PLATE_PATTERN.match(plate_text) and plate_text not in already_seen:
                                confidence = round(sum(scores) / len(scores), 3) if scores else None
                                plates[car_id] = {"plate": plate_text, "confidence": confidence}
                                print(f"Car {car_id}: {plate_text} ({confidence})")

                                yield {
                                    "type": "plate_found",
                                    "vehicle_id": car_id,
                                    "vehicle": f"Car {car_id}",
                                    "plate": plate_text,
                                    "confidence": confidence
                                }

                            break
                        break

        
        for car_id, (vx1, vy1, vx2, vy2) in boxes.items():
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
            if car_id in plates:
                cv2.putText(
                    frame, plates[car_id]["plate"], (vx1, max(vy1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                )

        if processed_idx % SEND_FRAME_EVERY_N_PROCESSED == 0:
            ok, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]
            )
            if ok:
                b64_frame = base64.b64encode(jpeg.tobytes()).decode("ascii")
                yield {"type": "frame", "image": b64_frame}

    cap.release()

    yield {"type": "done", "count": len(plates)}


_SENTINEL = object()


async def iterate_in_thread(generator_fn, *args):
    
    loop = asyncio.get_running_loop()
    q: "queue_module.Queue" = queue_module.Queue(maxsize=4)

    def worker():
        try:
            for item in generator_fn(*args):
                q.put(item)
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _SENTINEL:
            break
        yield item


@app.get("/")
def home():
    return {"message": "Number Plate Detection API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "No file selected"}

    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png")
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        return {"error": "Unsupported file format"}

    unique_name = str(uuid.uuid4()) + extension
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    print("Uploaded:", file_path)

    if extension in (".jpg", ".jpeg", ".png"):
        return {
            "message": "Image uploaded. Image detection can be added next.",
            "results": []
        }

    try:
        results = [
            update for update in process_video_streaming(file_path)
            if update.get("type") == "plate_found"
        ]
    except Exception as e:
        print("Detection error:", str(e))
        return {"error": str(e), "results": []}

    try:
        os.remove(file_path)
    except Exception:
        pass

    return {
        "message": "Detection completed",
        "filename": file.filename,
        "count": len(results),
        "results": results
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "No file selected"}

    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv")
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        return {"error": "Unsupported file format"}

    unique_name = str(uuid.uuid4()) + extension
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    print("Uploaded:", file_path)

    return {"file_id": unique_name}


@app.websocket("/ws/detect/{file_id}")
async def detect_ws(websocket: WebSocket, file_id: str):
    await websocket.accept()

    file_path = os.path.join(UPLOAD_FOLDER, file_id)

    if not os.path.isfile(file_path):
        await websocket.send_json({"type": "error", "message": "File not found"})
        await websocket.close()
        return

    try:
        async for update in iterate_in_thread(process_video_streaming, file_path):
            await websocket.send_json(update)

        try:
            os.remove(file_path)
        except Exception:
            pass

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Detection error (ws):", str(e))
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
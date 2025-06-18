from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from extractor import Extractor
import ultralytics
import utils
import requests
import os
import traceback
import base64

idcard_extractor = Extractor()
router = APIRouter()
model = ultralytics.YOLO("best.pt")

def resizeImage(img):
    return cv2.resize(img, (900, 600), interpolation=cv2.INTER_AREA)

def is_valid_image(img):
    return img is not None and isinstance(img, np.ndarray) and img.size > 0

def extract_points(result):
    points = {}
    for box in result.boxes:
        class_index = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if class_index in utils.class_to_corner:
            corner_name = utils.class_to_corner[class_index]
            points[corner_name] = (center_x, center_y)
    return points

def perspective_transform(img, points):
    required = ["top_left", "top_right", "bottom_right", "bottom_left"]
    if all(k in points and points[k] is not None for k in required):
        rect = np.array([
            points["top_left"], points["top_right"],
            points["bottom_right"], points["bottom_left"]
        ], dtype="float32")
        width = int(np.linalg.norm(rect[0] - rect[1]))
        height = int(np.linalg.norm(rect[0] - rect[3]))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (width, height))
    return img

@router.post("/ocr/front")
async def upload_front_only(id_card_front: UploadFile = File(...)):
    try:
        front_bytes = await id_card_front.read()
        front_array = np.asarray(bytearray(front_bytes), dtype=np.uint8)
        front_img = cv2.imdecode(front_array, cv2.IMREAD_COLOR)

        if not is_valid_image(front_img):
            return JSONResponse(status_code=400, content={"status_code": 400, "message": "Ảnh không hợp lệ", "data": None, "error": "Lỗi đọc ảnh"})

        result = model(front_img)[0]
        points = extract_points(result)
        front_img = perspective_transform(front_img, points)

        if not is_valid_image(front_img):
            raise ValueError("Ảnh mặt trước sau xử lý bị lỗi.")

        annotations = idcard_extractor.Detection(front_img)
        extracted_result = []
        
        if annotations is not None and len(annotations) > 0:
            extracted_result = [
                idcard_extractor.WarpAndRec(front_img, *box)
                for box in reversed(annotations)
            ]

        front_info = idcard_extractor.GetInformationFront(extracted_result)

        # if not os.path.exists("./tmp"):
            # os.makedirs("./tmp")

            # front_dir = f"./tmp/{front_info['identity_card_number']}_mattruoc.jpg"
            # cv2.imwrite(front_dir, resizeImage(front_img), [cv2.IMWRITE_JPEG_QUALITY, 95])

         # with open(front_dir, "rb") as f1:
             # files = [("files", (front_dir, f1, "image/jpeg"))]
            # response = requests.post(utils.url, files=files)

         # id_card_front = response.json().get("data", [None])[0] if response.status_code == 201 else None

        return JSONResponse(status_code=200, content={
            "status_code": 200,
            "message": "Trích xuất mặt trước CCCD thành công",
            "data": {**front_info, "id_card_front": None},
            "error": None
        })

    except Exception as e:
        traceback.print_exc()
        print("DEBUG error:", e)
        print("DEBUG type of error:", type(e))
        return JSONResponse(status_code=400, content={"status_code": 400, "message": "Không thể trích xuất thông tin CCCD", "data": None, "error": str(e)})

@router.post("/ocr")
async def upload_image(id_card_front: UploadFile = File(...), id_card_back: UploadFile = File(...)):
    try:
        front_img = cv2.imdecode(np.frombuffer(await id_card_front.read(), np.uint8), cv2.IMREAD_COLOR)
        back_img = cv2.imdecode(np.frombuffer(await id_card_back.read(), np.uint8), cv2.IMREAD_COLOR)

        if not is_valid_image(front_img) or not is_valid_image(back_img):
            raise ValueError("Ảnh CCCD bị lỗi hoặc không hợp lệ")

        results = model([front_img, back_img])
        images = []
        for img, result in zip([front_img, back_img], results):
            img = perspective_transform(img, extract_points(result))
            if not is_valid_image(img):
                raise ValueError("Ảnh CCCD sau xử lý bị lỗi")
            images.append(img)

        front_annotations = idcard_extractor.Detection(images[0])
        front_result = [idcard_extractor.WarpAndRec(images[0], *box) for box in reversed(front_annotations)] if front_annotations else []
        front_info = idcard_extractor.GetInformationFront(front_result)

        back_annotations = idcard_extractor.Detection(images[1])
        back_result = [idcard_extractor.WarpAndRec(images[1], *box) for box in reversed(back_annotations)] if back_annotations else []
        back_info = idcard_extractor.GetInformationBack(back_result)

        #if not os.path.exists("./tmp"):
            #os.makedirs("./tmp")

        #front_dir = f"./tmp/{front_info['identity_card_number']}_mattruoc.jpg"
        #back_dir = f"./tmp/{front_info['identity_card_number']}_matsau.jpg"
        #cv2.imwrite(front_dir, resizeImage(images[0]), [cv2.IMWRITE_JPEG_QUALITY, 95])
        #cv2.imwrite(back_dir, resizeImage(images[1]), [cv2.IMWRITE_JPEG_QUALITY, 95])

        #with open(front_dir, "rb") as f1, open(back_dir, "rb") as f2:
            #files = [("files", (front_dir, f1, "image/jpeg")), ("files", (back_dir, f2, "image/jpeg"))]
            #response = requests.post(utils.url, files=files)

        #id_card_front = response.json().get("data", [None, None])[0] if response.status_code == 201 else None
        #id_card_back = response.json().get("data", [None, None])[1] if response.status_code == 201 else None

        data = {
            **front_info,
            "id_card_issued_date": back_info.get("id_card_issued_date"),
            "id_card_front": None,
            "id_card_back": None,
        }

        #empty = [k for k, v in data.items() if not v]
        #if len(empty) > 3:
            #return JSONResponse(status_code=400, content={"status_code": 400, "message": "Không thể trích xuất đủ thông tin CCCD", "data": None, "error": f"Thiếu quá nhiều thông tin: {', '.join(empty)}"})

        return JSONResponse(status_code=200, content={"status_code": 200, "message": "Trích xuất thông tin CCCD thành công", "data": data, "error": None})

    except Exception as e:
        return JSONResponse(status_code=400, content={"status_code": 400, "message": "Không thể trích xuất thông tin CCCD", "data": None, "error": str(e)})
@router.post("/cut-cccd")
async def cutcccd(id_card_front: UploadFile = File(...)):
    # Read image from UploadFile
    contents = await id_card_front.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    class_to_corner = {0: "bottom_left", 1: "bottom_right", 4: "top_left", 5: "top_right"}

    # Run YOLO detection
    results = model(image)

    for result in results:
        points = {}
        for box in result.boxes:
            class_index = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0]

            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)

            if class_index in class_to_corner:
                corner_name = class_to_corner[class_index]
                points[corner_name] = (center_x, center_y)

        if all(k in points for k in ["top_left", "top_right", "bottom_right", "bottom_left"]):
            rect = np.array(
                [
                    points["top_left"],
                    points["top_right"],
                    points["bottom_right"],
                    points["bottom_left"],
                ],
                dtype="float32"
            )
            width = int(np.linalg.norm(rect[0] - rect[1]))
            height = int(np.linalg.norm(rect[0] - rect[3]))
            dst = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32"
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (width, height))

            # Encode to Base64
            _, buffer = cv2.imencode('.jpg', warped)
            b64_string = base64.b64encode(buffer).decode('utf-8')

            return JSONResponse(content={"base64": b64_string})

        else:
            return JSONResponse(content={"error": "Not all corners detected."}, status_code=400)

    return JSONResponse(content={"error": "No results from model."}, status_code=500)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True)

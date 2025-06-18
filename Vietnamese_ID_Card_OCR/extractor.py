import os
import re
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import utils

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ocr = None
detector = None


class Extractor:

    def __init__(self):
        self.config = Cfg.load_config_from_name("vgg_seq2seq")
        self.config["weights"] = "./weights/seq2seqocr.pth"
        # self.config = Cfg.load_config_from_name("vgg_transformer")
        # self.config["weights"] = "./weights/transformerocr.pth"
        self.config["cnn"]["pretrained"] = False
        self.config["device"] = "cpu"
        if ocr == None:
            self.ocr = PaddleOCR(
                lang="en",
                ocr_version="PP-OCRv4",
                use_space_char=True,
            )
            print("OCR error: khoi tao")
        else:
            self.ocr = ocr
            print("OCR error: chhay luon")
        if detector == None:
            self.detector = Predictor(self.config)
        else:
            self.detector = detector

    def Detection(self, frame):
        annotations = self.ocr.ocr(frame, rec=False, cls=False)
        return annotations[0]
    # def Detection(self, frame):
        # try:
            # result = self.ocr.ocr(frame, rec=False, cls=False)
            # # result là list [[box1], [box2], ...]
            # if isinstance(result, list) and len(result) > 0 and len(result[0]) > 0:
                # annotations = result
            # else:
                # annotations = []
        # except Exception as e:
            # print("OCR error:", e)
            # annotations = []

        # return annotations


    def WarpAndSave(
        self, frame, fileName, top_left, top_right, bottom_right, bottom_left
    ):

        w, h, cn = frame.shape
        padding = 4.0
        padding = int(padding * w / 640)

        # All points are in format [cols, rows]
        pt_A = top_left[0], top_left[1]
        pt_B = bottom_left[0], bottom_left[1]
        pt_C = bottom_right[0], bottom_right[1]
        pt_D = top_right[0], top_right[1]

        # Here, I have used L2 norm. You can use L1 also.
        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32(
            [
                [0, 0],
                [0, maxHeight - 1],
                [maxWidth - 1, maxHeight - 1],
                [maxWidth - 1, 0],
            ]
        )

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        matWarped = cv2.warpPerspective(
            frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR
        )
        cv2.imwrite(fileName, matWarped)

        return True

    def WarpAndRec(self, frame, top_left, top_right, bottom_right, bottom_left):
        w, h, cn = frame.shape
        padding = 4.0
        padding = int(padding * w / 640)

        box = []
        # All points are in format [cols, rows]
        pt_A = top_left[0] - padding, top_left[1] - padding
        pt_B = bottom_left[0] - padding, bottom_left[1] + padding
        pt_C = bottom_right[0] + padding, bottom_right[1] + padding
        pt_D = top_right[0] + padding, top_right[1] - padding

        # Here, I have used L2 norm. You can use L1 also.
        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32(
            [
                [0, 0],
                [0, maxHeight - 1],
                [maxWidth - 1, maxHeight - 1],
                [maxWidth - 1, 0],
            ]
        )

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        matWarped = cv2.warpPerspective(
            frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR
        )

        s = self.detector.predict(Image.fromarray(matWarped))

        box.append(pt_A)
        box.append(pt_D)
        box.append(pt_C)
        box.append(pt_B)

        return [s, box]

    def GetInformationFront(self, _results):

        result = {}
        result["identity_card_number"] = ""
        result["full_name"] = ""
        result["date_of_birth"] = ""
        result["gender"] = ""
        result["nationality"] = "Việt Nam"
        result["place_of_origin"] = ""
        result["place_of_residence"] = ""
        result["id_card_expired_date"] = ""

        for i, res in enumerate(_results):
            s = res[0]
            #print(s)
            if re.search(utils.regex_id_number, s) and (
                not result["identity_card_number"]
            ):
                result["identity_card_number"] = s
                continue

            if re.search(r"Ho|va|tên|ten|Full|name", s):

                if (
                    not re.search(r"[0-9]", _results[i + 1][0])
                    and _results[i + 1][0].isupper()
                ):
                    name = _results[i + 1]
                else:
                    name = _results[i + 2]

                result["full_name"] = name[0].title()

                continue

            if re.search(r"Ngay|sinh|birth|bith", s) and not result["date_of_birth"]:
                # Check if date is embedded with other text
                if re.search(utils.regex_dob, s):
                    dob = re.sub(r"[^0-9/]", "", s)
                    result["date_of_birth"] = dob = (
                        dob[1:].strip() if dob.startswith("/") else dob.strip()
                    )
                elif re.search(utils.regex_dob, _results[i - 1][0]):
                    dob = re.sub(r"[^0-9/]", "", _results[i - 1][0])
                    result["date_of_birth"] = dob = (
                        dob[1:].strip() if dob.startswith("/") else dob.strip()
                    )
                elif re.search(utils.regex_dob, _results[i + 1][0]):
                    dob = re.sub(r"[^0-9/]", "", _results[i + 1][0])
                    result["date_of_birth"] = dob = (
                        dob[1:].strip() if dob.startswith("/") else dob.strip()
                    )
                else:
                    dob = []

                continue

            if re.search(r"giá|trị|expiry", s) and not result["id_card_expired_date"]:
                # Check if date is embedded with other text
                if re.search(utils.regex_dob, s):
                    dob = re.sub(r"[^0-9/]", "", s)
                    result["id_card_expired_date"] = dob.strip()
                elif re.search(utils.regex_dob, _results[i - 1][0]):
                    dob = re.sub(r"[^0-9/]", "", _results[i - 1][0])
                    result["id_card_expired_date"] = dob.strip()

                elif re.search(utils.regex_dob, _results[i + 1][0]):
                    dob = re.sub(r"[^0-9/]", "", _results[i + 1][0])
                    result["id_card_expired_date"] = dob.strip()
                else:
                    dob = []

                continue

            if re.search(r"Giới|Gioi|Sex", s):
                gender = _results[i]
                result["gender"] = (
                    "FEMALE" if re.search(r"Nữ|nữ|Nu|nu", gender[0]) else "MALE"
                )
                continue

            if re.search(r"Quê|Que|origin|ongin|ngin|orging", s):
                if not re.search(utils.regex_residence, _results[i + 1][0]):
                    place_of_origin = [_results[i], _results[i + 1]]
                else:
                    place_of_origin = []

                if place_of_origin:
                    if (
                        len(
                            re.split(":|;|of|ging|gin|ggong", place_of_origin[0][0])[
                                -1
                            ].strip()
                        )
                        > 2
                    ):
                        result["place_of_origin"] = (
                            (re.split(":|;|of|ging|gin|ggong", place_of_origin[0][0]))[
                                -1
                            ].strip()
                            + ", "
                            + place_of_origin[1][0]
                        )
                    else:
                        result["place_of_origin"] = place_of_origin[1][0]

                continue

            if re.search(r"Nơi|Noi|tru|thuong|trú|residence", s):
                if not re.search(utils.regex_residence, _results[i + 1][0]):
                    place_of_residence = [_results[i], _results[i + 1]]
                elif not re.search(utils.regex_residence, _results[-1][0]):
                    place_of_residence = [_results[i], _results[-1]]
                elif not re.search(utils.regex_residence, _results[-2][0]):
                    place_of_residence = [_results[i], _results[-2]]
                if place_of_residence:
                    first_line = re.split(
                        ":|;|of|residence|ence|end", place_of_residence[0][0]
                    )[-1].strip()
                    if len(first_line) > 2:
                        result["place_of_residence"] = (
                            first_line + ", " + place_of_residence[1][0]
                        )
                    else:
                        result["place_of_residence"] = place_of_residence[1][0]

                continue

            else:
                continue

        return result

    def GetInformationBack(self, _results):

        result = {}
        result["id_card_issued_date"] = ""
        #print(_results)
        for i, res in enumerate(_results):
            s = res[0]
            if (
                re.search(r"Date|month|year|Ngày|tháng|năm", s)
                and not result["id_card_issued_date"]
            ):
                # Check if date is embedded with other text
                if re.search(utils.regex_dob, s):
                    dob = re.sub(r"[^0-9/]", "", s)
                    result["id_card_issued_date"] = dob = (
                        dob[1:].strip() if dob.startswith("/") else dob.strip()
                    )
                elif re.search(utils.regex_dob, _results[i - 1][0]):
                    dob = re.sub(r"[^0-9/]", "", _results[i - 1][0])
                    result["id_card_issued_date"] = dob = (
                        dob[1:].strip() if dob.startswith("/") else dob.strip()
                    )
                elif re.search(utils.regex_dob, _results[i + 1][0]):
                    dob = re.sub(r"[^0-9/]", "", _results[i + 1][0])
                    result["id_card_issued_date"] = dob = (
                        dob[1:].strip() if dob.startswith("/") else dob.strip()
                    )
                else:
                    dob = []

                continue

        return result
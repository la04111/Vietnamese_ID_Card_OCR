import paddleocr
import os

# Tìm đúng file paddleocr.py
paddleocr_path = os.path.dirname(paddleocr.__file__)
target_file = os.path.join(paddleocr_path, "paddleocr.py")

if not os.path.exists(target_file):
    print(f"❌ Không tìm thấy file: {target_file}")
else:
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Sửa tất cả các dòng `if not dt_boxes:` thành kiểm tra rõ ràng
    patched = content.replace(
        'if not dt_boxes:',
        'if dt_boxes is None or len(dt_boxes) == 0:'
    )

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(patched)

    print("✅ Đã sửa lỗi liên quan đến `dt_boxes` trong paddleocr.py")

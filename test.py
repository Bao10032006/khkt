import cv2
import numpy as np
diem = [175.5, 247.0]
mang_diem=[(320, 65), (320, 206), (360, 195), (355, 55)]
def kiem_tra_diem_trong_mang(diem, mang_diem):
    # Chuyển đổi mảng điểm thành một mảng numpy với kiểu dữ liệu phù hợp
    mang_diem = np.array(mang_diem, dtype=np.int32)
    diem = tuple(diem)

    # Sử dụng hàm cv2.pointPolygonTest để kiểm tra
    # Hàm trả về giá trị 1 nếu điểm nằm trong đa giác, 0 nếu điểm nằm trên cạnh và -1 nếu nằm ngoài
    
    return cv2.pointPolygonTest(mang_diem, diem, False) >= 0

kiem_tra_diem_trong_mang(diem,mang_diem)
import cv2
import numpy as np
import sys

# Baca gambar
img = cv2.imread(sys.path[0]+"/lingkaran.jpg", cv2.IMREAD_GRAYSCALE)

# pre-processing
imgm = cv2.medianBlur(img,5)
imgb = cv2.cvtColor(imgm, cv2.COLOR_GRAY2BGR)


# Deteksi lingkaran menggunakan Hough Transform
circles = cv2.HoughCircles(
    imgm,
    cv2.HOUGH_GRADIENT,
    1,  # Resolusi ruang akumulator yang diinginkan (semakin kecil semakin akurat, tapi juga membutuhkan lebih banyak waktu)
    50,  # Jarak minimum antara pusat dua lingkaran yang dideteksi
    param1=100,  # Parameter deteksi tepi (semakin tinggi, semakin ketat)
    param2=50,  # Parameter ambang batas untuk memilih pusat lingkaran (semakin kecil, semakin ketat)
    minRadius=0,  # Radius minimum lingkaran yang akan dideteksi
    maxRadius=100  # Radius maksimum lingkaran yang akan dideteksi
)

# Jika lingkaran ditemukan, gambar mereka pada gambar asli
if circles is not None:
   # cari koordinat x,y dan radius (r)
   circles = np.round(circles[0, :]).astype("int")
   print(circles)
   # cari terus dlm loop
   for (x, y, r) in circles:
      cv2.circle(imgb, (x, y), r, (0, 255, 0), 2)

# Tampilkan hasil
cv2.imshow('Deteksi Lingkaran', imgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

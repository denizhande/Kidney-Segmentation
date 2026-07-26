import os
import cv2
import numpy as np
import nibabel as nib
from matplotlib.pyplot import imsave

# Kaydetme yolları
image_output_dir = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\2d_data\\images"
mask_output_dir = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\2d_data\\masks"

# Eğer klasörler yoksa oluştur
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# KiTS19 veri setini yükleme ve işleme
def load_kits19_data(data_path, input_size=(128, 128)):
    cases = os.listdir(data_path)

    for case in cases:
        case_path = os.path.join(data_path, case)
        image_path = os.path.join(case_path, "imaging.nii.gz")
        mask_path = os.path.join(case_path, "segmentation.nii.gz")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping case {case}: Required files not found.")
            continue

        # NIfTI dosyalarını yükle
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Her dilimi işle
        for i in range(image.shape[2]):  # Z ekseni boyunca her dilimi işle
            img_slice = image[:, :, i]  # Görüntü dilimi
            mask_slice = mask[:, :, i]  # Maske dilimi

            # Eğer maske tamamen sıfırsa kaydetme
            if np.all(mask_slice == 0):
                print(f"Skipping slice {i} of case {case}: Mask is empty.")
                continue

            # Yeniden boyutlandır
            img_slice_resized = cv2.resize(img_slice, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
            mask_slice_resized = cv2.resize(mask_slice, (input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST)

            # Kaydetme dosya adlarını oluştur
            image_file_path = os.path.join(image_output_dir, f"{case}_image_slice_{i}.png")
            mask_file_path = os.path.join(mask_output_dir, f"{case}_mask_slice_{i}.png")

            # Görüntü ve maske dilimlerini kaydet
            imsave(image_file_path, img_slice_resized, cmap='gray')
            imsave(mask_file_path, mask_slice_resized, cmap='gray')

            print(f"Görüntü kaydedildi: {image_file_path}")
            print(f"Maske kaydedildi: {mask_file_path}")

# Veri yolu
data_path = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\data"

# İşlemi başlat
load_kits19_data(data_path, input_size=(128, 128))



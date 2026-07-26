import matplotlib.pyplot as plt
from keras.src.saving import load_model
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from sklearn.metrics import accuracy_score,jaccard_score,precision_score,recall_score
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# Görüntüleri ve maskeleri yükleme
def load_data(image_dir, mask_dir, input_size=(128, 128)):
    images = []
    masks = []

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Görüntü ve maske yükle
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Skipping file pair: {img_file}, {mask_file}")
            continue

        # Yeniden boyutlandır
        img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, input_size, interpolation=cv2.INTER_NEAREST)

        # Görüntüyü (H, W, C) boyutuna, maske ise (H, W, 1) boyutuna dönüştür
        images.append(np.expand_dims(img_resized, axis=-1))  # Görüntü (H, W, 1)
        masks.append(np.expand_dims(mask_resized, axis=-1))  # Maske (H, W, 1)

    return np.array(images), np.array(masks)

image_path = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\2d_data\\images"
mask_path = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\2d_data\\masks"
output_size = (128, 128)

images, masks = load_data(image_path, mask_path, input_size=output_size)

# Verileri float32 türüne dönüştür
images = images.astype(np.float32) / 255.0  # Görüntüleri [0,1] aralığına normalize et
masks = masks.astype(np.float32) / 255.0  # Maskeleri [0,1] aralığına normalize et

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

#model=load_model("deeplab_kidney_model_2d.h5")
#test_loss, test_accuracy, test_dice = model.evaluate(X_test, y_test)
#print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

model = tf.keras.models.load_model(str('C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\deeplab_kidney_model_2d_v8_80e.h5'), compile=False)

dice_scores= []                                                              # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
acc_scores= []                                                               # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
pre_scores= []                                                               # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
rec_scores= []                                                               # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
ji_scores= []                                                                # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
                                                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
for k, img in enumerate(X_test):                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    test = np.expand_dims(img, axis=0)                                       # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    ground_truth = y_test[k]                                                 # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    predict = model.predict(test)                                            # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
                                                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    # plt.imshow(predict[0,:,:,0] > 0.5, cmap=plt.cm.gray)                   # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    # plt.show()                                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
                                                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    dice = dice_coef(ground_truth, predict > 0.5)                            # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    y_true = np.array(ground_truth, dtype='bool').flatten()                  # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    y_pred = np.array(predict > 0.5).flatten()                               # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    accuracy = accuracy_score(y_true, y_pred)                                # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    jaccard = jaccard_score(y_true, y_pred)                                  # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    precision = precision_score(y_true, y_pred, zero_division=1)             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    rec = recall_score(y_true, y_pred)                                       # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
                                                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    dice_scores.append(dice)                                                 # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    acc_scores.append(accuracy)                                              # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    pre_scores.append(precision)                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    rec_scores.append(rec)                                                   # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
    ji_scores.append(jaccard)                                                # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
                                                                             # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.                                                         # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
print("mean DSC:", np.mean(dice_scores))                                     # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
print("mean ACC",np.mean(acc_scores))                                        # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
print("mean PRE", np.mean(pre_scores))                                       # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
print("mean REC", np.mean(rec_scores))                                       # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.
print("mean JI", np.mean(ji_scores))                                         # Metrikleri burada hesaplıyor. plt olanları açarsan bütün test verisini çizdirir.


########## Görselleştirmek istediğin tek bir örnek için kod

sample_index = 4                                              # Kaçıncı örneği görselleştirmek istiyorsan onun indexi
img_to_visualize = X_test[sample_index]                       # X_test setinde o örneği çeken kod
test = np.expand_dims(img_to_visualize, axis=0)
ground_truth = y_test[sample_index]                           # Belirlediğin örneğe denk gelen maske
predict = model.predict(test)

# plt.imshow(predict[0,:,:,0] > 0.1, cmap=plt.cm.gray)
# plt.show()
                                                                               # Yukarıdaki metrik hesaplama çok uzun sürüyor.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))                   # Bir kere kaydettikten sonra kapatabilirsin ya da salla hiç hesaplama lazım değilse
axes[0].imshow(ground_truth, cmap=plt.cm.gray)
axes[0].set_title('Ground Truth')
axes[0].axis('off')
axes[1].imshow(predict[0,:,:,0] > 0.5, cmap=plt.cm.gray)
axes[1].set_title('Predicted Mask')
axes[1].axis('off')
plt.show()
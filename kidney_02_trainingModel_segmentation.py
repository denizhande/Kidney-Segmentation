import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric


# Dice skorunu hesaplayan özel metrik fonksiyonu
class DiceCoefficient(Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Eşikleme işlemi: maske değerini [0,1] aralığına getir
        y_true = tf.round(y_true)
        y_pred = tf.round(y_pred)

        # Kesişen alanı ve toplam alanı hesapla
        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        # Dice skoru hesaplama
        dice_score = (2.0 * intersection) / (total + tf.keras.backend.epsilon())

        # Sonuçları güncelle
        self.total.assign_add(dice_score)
        self.count.assign_add(1.0)

    def result(self):
        # Ortalamayı döndür
        return self.total / self.count

    def reset_state(self):
        # Durumları sıfırla
        self.total.assign(0)
        self.count.assign(0)


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


# DeepLabV3+ Modeli
def create_deeplab_model(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Encoder kısmı (özellik çıkarımı)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Decoder kısmı (boyut artırma)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)  # Boyut artırma
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)  # Boyut artırma

    # Sonuç katmanı (Çıkış boyutu 128x128)
    x = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)  # 128x128 çıkışı

    model = models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', DiceCoefficient()])
    return model


# Veri yükleme ve tür dönüşümü
image_path = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\2d_data\\images"
mask_path = "C:\\Users\\DELL XPS 8910\\PycharmProjects\\pythonProject\\kits19\\2d_data\\masks"
output_size = (128, 128)

images, masks = load_data(image_path, mask_path, input_size=output_size)

# Verileri float32 türüne dönüştür
images = images.astype(np.float32) / 255.0  # Görüntüleri [0,1] aralığına normalize et
masks = masks.astype(np.float32) / 255.0  # Maskeleri [0,1] aralığına normalize et

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# DeepLabV3+ Modeli oluşturma
deeplab_model = create_deeplab_model(input_shape=(128, 128, 1))

# Modeli eğitme
history = deeplab_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Modeli kaydet
deeplab_model.save("deeplab_kidney_model_2d.h5")

# Eğitim sırasında Dice Skoru eklemek
print(f"Training Dice Scores per Epoch:")
for epoch, dice_score in enumerate(history.history['dice_coefficient']):
    print(f"Epoch {epoch + 1}: Training Dice Score = {dice_score:.4f}")

# Test setini değerlendirme
test_loss, test_accuracy, test_dice = deeplab_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")

# np.array(X_train, dtype=np.float32)
# np.array(X_test, dtype=np.float32)
# np.array(y_train, dtype=np.float32)
# np.array(y_test, dtype=np.float32)
# DeepLabV3+ Modeli oluşturma





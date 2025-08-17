import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SAVE_PATH = os.path.join("saved_model", "pollen_model.h5")
IMG_SIZE = (128, 128)

def build_model(num_classes=23):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),

        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.2),
        Dense(500, activation='relu'),
        Dense(150, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory("dataset/", target_size=IMG_SIZE,
                                            batch_size=32, subset='training')
    val_gen = datagen.flow_from_directory("dataset/", target_size=IMG_SIZE,
                                          batch_size=32, subset='validation')

    model = build_model(num_classes=train_gen.num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    os.makedirs("saved_model", exist_ok=True)
    model.save(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

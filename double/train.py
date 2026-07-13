import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dataset_path = r"C:\Users\nandh\OneDrive\Desktop\two digit\double\double_digit_dataset"
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.15,

    brightness_range=(0.8,1.2),

    fill_mode="nearest"
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64,128),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    subset="training",
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64,128),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    subset="validation",
    shuffle=False
)

model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(64,128,1)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(100,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=25,
    callbacks=[checkpoint,earlystop]
)

print("Training Completed Successfully")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# create model folder automatically
os.makedirs("model", exist_ok=True)

# dataset paths
train_path = "dataset/train"
val_path = "dataset/validation"

# image generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

print("Classes:", train_data.class_indices)

# MobileNetV2 base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# save model automatically
model.save("model/healthy_vs_rotten.h5")

print("✅ Model saved at model/healthy_vs_rotten.h5")
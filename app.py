from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# load model
MODEL_PATH = "model/healthy_vs_rotten.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# class names from dataset folder
classes = sorted(os.listdir("dataset/train"))
print("Classes:", classes)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET","POST"])
def upload():

    if request.method == "POST":

        file = request.files["file"]

        if file and file.filename != "":

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # preprocess image
            img = image.load_img(filepath, target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # prediction
            prediction = model.predict(img_array)

            index = np.argmax(prediction)
            confidence = round(float(np.max(prediction))*100,2)

            result = classes[index]

            # if others → invalid image and set confidence 100%
            if result == "others":
                result = "Invalid Image (Not Fruit/Vegetable)"
                confidence = 100.00

            return render_template(
                "result.html",
                result=result,
                confidence=confidence,
                image_path=filepath
            )

    return render_template("upload.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")
if __name__ == "__main__":
    app.run(debug=True)
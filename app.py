from flask import Flask, render_template_string, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("brain_tumor_model.h5")
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

HTML = """
<html><body style="text-align:center; font-family:Arial; margin-top:50px">
<h1>Brain Tumor Detection - Bhavani Project</h1>
<form method="post" enctype="multipart/form-data">
<input type="file" name="file"><br><br>
<input type="submit" value="Predict">
</form>
<h2 style="color:green">{{result}}</h2>
</body></html>
"""

@app.route('/', methods=['GET','POST'])
def home():
    result = ""
    if request.method == 'POST':
        f = request.files['file']
        f.save("temp.jpg")
        img = image.load_img("temp.jpg", target_size=(150,150))
        img = image.img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        result = "Prediction: " + classes[np.argmax(pred)]
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
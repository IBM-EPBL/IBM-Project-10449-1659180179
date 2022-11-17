import logging
import os
import pickle

import cv2
import sklearn
import matplotlib
import numpy as np
matplotlib.use('Agg')
import tensorflow
from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils import grab_contours, sort_contours, label_contour, resize_image
import matplotlib.pyplot as plt

plt.axis("off")

# Kernel for OpenCV morphological operations
kernel = np.ones((1,1),np.uint8)

# Disable Tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Folder to store images
IMAGE_FOLDER = 'static/images'

app = Flask(__name__)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Set Tensorflow to use CPU
tensorflow.config.set_visible_devices([], 'GPU')

# Load the model
model = load_model("model/model.h5")

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        filepath = (f.filename)
        # Save the image
        f.save(os.path.join(app.config['IMAGE_FOLDER'], filepath))

        upload_img = os.path.join(IMAGE_FOLDER, filepath)

        image = cv2.imread(upload_img)  # Read the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray = cv2.GaussianBlur(gray, (3, 3), 0)    # Noise Removal
        edged = cv2.Canny(gray, 40, 120)    # Detect edges
    
        # Isolate individual digits
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cnts = grab_contours(cnts)
        cnts = sort_contours(cnts, method='left-to-right')[0]
         
        extracted_digits = []
        i = 0
        # Loop over the detected contours
        for c in cnts:
            # Compute bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)    

            # Filter extracted bounding boxes based on size criteria to avoid processing unwanted contours
            if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
                i += 1               
                # Extract the character
                roi = gray[y:y + h, x:x + w]
                
                # Perform image inverting it to make the digit appear as *white* (foreground) on a *black* background
                thresh = cv2.bitwise_not(roi)
                
                # Resize the image to 28x28 pixels
                thresh = resize_image(thresh)                
                
                # Noise removal before feeding to the model
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.GaussianBlur(thresh, (3,3), 0)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # Normalize pixel values between 0 and 1
                thresh = thresh / 255.0
                
                plt.imshow(thresh, cmap='gray')
                plt.savefig('static/images/digits/digit_' + str(i))
                
                # Add to list of extracted individual digits
                extracted_digits.append(thresh)

        # Plot uploaded image with bounding boxes around individual digits
        bbox_path = os.path.join(app.config['IMAGE_FOLDER'], 'bbox.png')
        plt.imshow(image, cmap = 'gray')
        plt.savefig(bbox_path)

                
        # Make predictions on the extracted digits and display the output
        predictions = []
        if len(extracted_digits) == 0:
            return render_template('predict.html', num='No digits detected')
        else:
            extracted_digits = np.array(extracted_digits)
            predictions = model.predict(extracted_digits)
            predictions = np.argmax(predictions, axis=1)
            print("Predictions ", predictions)
        
        return render_template('predict.html', num=predictions, img_path = bbox_path)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
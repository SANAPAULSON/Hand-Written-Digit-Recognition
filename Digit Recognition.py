import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)

# GUI function to predict the handwritten digit
def predict_digit():
    global model
    img = PIL_image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    accuracy = np.max(prediction) * 100
    prediction_label.config(text=f"Predicted Digit: {digit}\nAccuracy: {accuracy:.2f}%")

# GUI setup
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = Canvas(root, width=200, height=200, bg='white')
canvas.pack()

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='white', width=5)
    draw.line([x1, y1, x2, y2], fill='white', width=5)

def clear_canvas():
    canvas.delete("all")
    global PIL_image, draw
    PIL_image = Image.new("RGB", (200, 200), "black")
    draw = ImageDraw.Draw(PIL_image)

canvas.bind("<B1-Motion>", paint)
clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack()
predict_button = Button(root, text="Predict Digit", command=predict_digit)
predict_button.pack()
prediction_label = Label(root, text="")
prediction_label.pack()

# Initialize PIL image and draw object
PIL_image = Image.new("RGB", (200, 200), "black")
draw = ImageDraw.Draw(PIL_image)

root.mainloop()
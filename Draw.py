import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# load that other model into this
model = tf.keras.models.load_model('mnist_model.keras')

canvas_size = 280
image_size = 28

# tkinter window
root = tk.Tk()
root.title("Draw a single digit number")

# L is the image mode, representing greyscale here
# 0 is the initial color of everything (black). It matches mnist data I think
image = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(image)

# tkinter display canvas setup
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, background="black",
                   bd=2, relief="solid")
canvas.pack(pady=10)


# Draw function
def draw_digit(event):
    # testing
    print(f"drawing at {event.x},{event.y}")

    x, y = event.x, event.y
    r = 5  # brush radius. might change later

    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    # ^this is for the canvas popup, and V this is for the actual PIL image file
    draw.ellipse((x-r, y-r, x+r, y+r), fill=255)


# this means left mouse (B1) click and drag (Motion) will use the function above
canvas.bind("<B1-Motion>", draw_digit)

# prediction label
result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 16))
result_label.pack()


def predict_digit():
    image_resized = image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)

    # makes a 28x28 array of the image and its pixel values from 0-1
    img_array = np.array(image_resized) / 255.0
    # this reshapes it into (1,28,28,1)
    img_reshaped = img_array.reshape(1, image_size, image_size, 1).astype(np.float32)

    # creates a vector of probabilities, with a 0.X for each possible digit 0-9
    prediction = model.predict(img_reshaped)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # this updates the existing labels text, we don't need a whole new label
    # .config changes a widgets properties AFTER it was alr created
    result_label.config(text=f"Prediction {digit} ({confidence: .2%})")


def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_size, canvas_size), fill=0)

    root.update_idletasks()  # force redraw


# all the buttons
btn_predict = tk.Button(root, text="Predict", command=predict_digit)
btn_predict.pack()

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack()

root.mainloop()

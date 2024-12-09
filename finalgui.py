import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox, Toplevel
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.distance import cdist
import warnings
import threading
import time

warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# Load the Pretrained Model and Data
# -------------------------
model = load_model('vgg16_feature_extractor.h5')
features_matrix = np.load('features_matrix.npy')
dataset_df = pd.read_csv('dataset_df.csv')

print("Model, features_matrix, and dataset_df loaded successfully.")

def extract_embedding(image_path, target_size=(224,224)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    return features

root = tk.Tk()
root.title("Morphological Matching for Similar Image Detection")
root.geometry("900x600")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Morphological Matching for Similar Image Detection", 
                       font=("Arial", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

top_frame = tk.Frame(root, bg="#f0f0f0")
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

selected_image_path = None
image_display = None

middle_frame = tk.Frame(root, bg="#f0f0f0")
middle_frame.pack(pady=10)

image_label = tk.Label(middle_frame, bg="#f0f0f0")
image_label.pack(pady=5)

status_label = tk.Label(middle_frame, text="Ready. Please load an image.", font=("Arial", 10), bg="#f0f0f0")
status_label.pack()

output_frame = tk.Frame(root, bg="#f0f0f0")
output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

images_container = tk.Frame(output_frame, bg="#f0f0f0")
images_container.pack(side=tk.TOP, fill=tk.X)

bottom_frame = tk.Frame(output_frame, bg="#f0f0f0")
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

output_text = tk.Text(bottom_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(bottom_frame, command=output_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
output_text.configure(yscrollcommand=scrollbar.set)

def load_image():
    global selected_image_path, image_display
    file_path = filedialog.askopenfilename(
        title="Select an Image", 
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if file_path:
        selected_image_path = file_path
        img = Image.open(selected_image_path)
        img.thumbnail((300,300))
        image_display = ImageTk.PhotoImage(img)
        image_label.config(image=image_display)
        image_label.image = image_display
        status_label.config(text=f"Selected Image: {os.path.basename(selected_image_path)}")
    else:
        status_label.config(text="No image selected.")

def clear_output():
    global selected_image_path
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.config(state=tk.DISABLED)
    image_label.config(image='')
    image_label.image = None
    selected_image_path = None
    status_label.config(text="Ready. Please load an image.")
    for widget in images_container.pack_slaves():
        widget.destroy()

spinner_chars = ["|", "/", "-", "\\"]
stop_animation = False

def animate_processing(spinner_label):
    i = 0
    while True:
        if stop_animation:
            break
        spinner_label.config(text=spinner_chars[i % len(spinner_chars)])
        i += 1
        time.sleep(0.1)

def run_computation(query_feat, results_holder, error_holder):
    try:
        distances = cdist(query_feat, features_matrix, metric='cosine').flatten()
        dataset_df['distance'] = distances
        res = dataset_df.sort_values(by='distance').head(10)
        results_holder.append(res)
    except Exception as e:
        error_holder.append(str(e))

def finalize_results(results, processing_win):
    global stop_animation
    # Stop the animation
    stop_animation = True
    # Close the processing window
    processing_win.destroy()

    # Update UI with results
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, "Top 10 similar images:\n\n")
    for idx, row in results.iterrows():
        output_text.insert(tk.END, f"Image: {row['image_path']}, Label: {row['label']}, Distance: {row['distance']:.4f}\n")
    output_text.config(state=tk.DISABLED)

    # Clear old images frame
    for widget in images_container.pack_slaves():
        widget.destroy()

    images_frame = tk.Frame(images_container, bg="#f0f0f0")
    images_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
    images_frame.image_refs = []

    for idx, row in results.iterrows():
        img_path = row['image_path']
        if os.path.exists(img_path):
            pil_img = Image.open(img_path)
            pil_img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(pil_img)
            images_frame.image_refs.append(img_tk)
            img_label = tk.Label(images_frame, image=img_tk, bg="#f0f0f0")
            img_label.pack(side=tk.LEFT, padx=5)
        else:
            print(f"Warning: Image not found at {img_path}")

def submit():
    global selected_image_path, stop_animation
    if not selected_image_path:
        messagebox.showwarning("No Image", "Please load an image before submitting.")
        return

    query_feat = extract_embedding(selected_image_path)
    if query_feat is None:
        messagebox.showerror("Error", "Could not process the image. Try another one.")
        return

    # Create a processing overlay
    processing_win = Toplevel(root)
    processing_win.title("Processing")
    processing_win.geometry("300x100")
    processing_win.configure(bg="#f0f0f0")
    processing_win.grab_set()
    processing_label = tk.Label(processing_win, text="Your AI model is getting the required output...", bg="#f0f0f0", font=("Arial", 10))
    processing_label.pack(pady=10)

    spinner_label = tk.Label(processing_win, text="|", font=("Arial", 20), bg="#f0f0f0")
    spinner_label.pack(pady=5)

    results_holder = []
    error_holder = []
    stop_animation = False

    anim_thread = threading.Thread(target=animate_processing, args=(spinner_label,), daemon=True)
    anim_thread.start()

    comp_thread = threading.Thread(target=run_computation, args=(query_feat.reshape(1,-1), results_holder, error_holder), daemon=True)
    comp_thread.start()

    def check_thread():
        if comp_thread.is_alive():
            root.after(100, check_thread)
        else:
            # Computation done, wait 3 seconds before showing results
            if error_holder:
                # Stop animation right away if there's an error
                global stop_animation
                stop_animation = True
                processing_win.destroy()
                messagebox.showerror("Error", f"An error occurred: {error_holder[0]}")
                return

            # Computation succeeded, now wait 3 seconds
            # Keep animation running during these 3 seconds
            results = results_holder[0]
            root.after(3000, finalize_results, results, processing_win)

    check_thread()

load_button = tk.Button(top_frame, text="Load Image", command=load_image, bg="#e0e0e0", font=("Arial", 12))
load_button.pack(side=tk.LEFT, padx=5)

submit_button = tk.Button(top_frame, text="Submit", command=submit, bg="#c2f0c2", font=("Arial", 12, "bold"))
submit_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(top_frame, text="Clear Output", command=clear_output, bg="#f2b0b0", font=("Arial", 12, "bold"))
clear_button.pack(side=tk.LEFT, padx=5)

root.mainloop()

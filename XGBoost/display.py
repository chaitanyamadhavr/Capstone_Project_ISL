""" #Finalised on 6th October 2023.
import tkinter as tk
from PIL import Image, ImageTk
from googletrans import Translator
import threading

# Initialize the tkinter window
root = tk.Tk()
root.title("Sign Language Recognition")

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display text with borders and bigger font size
text_label = tk.Label(root, font=("Helvetica", 20), borderwidth=2, relief="solid")
text_label.pack()

translator = Translator()

# Function to translate English text to Kannada
def translate_english_to_kannada(text):
    try:
        translated = translator.translate(text, src='en', dest='kn').text
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation error"

prev = "Bois"
kan="prev kan"
# Function to update the window with a new frame and text
def update_window(frame, eng_text):
    global prev
    global kan
    # Create a PhotoImage object directly from the frame
    frame_img = ImageTk.PhotoImage(Image.fromarray(frame))

    # Update the image label with the new frame
    image_label.config(image=frame_img)
    image_label.image = frame_img

    # Translate English text to Kannada
    if(prev!=eng_text):
        kannada_text = translate_english_to_kannada(eng_text)
        prev=eng_text
        kan = kannada_text
    else:
        kannada_text = kan

    # Update the text label with the translated text
    text_label.config(text=kannada_text)

    root.update()  # Update the tkinter window

# Function to start the tkinter main loop in a separate thread
def start_tkinter_mainloop():
    root.mainloop()

# Create a thread to run the tkinter main loop
tkinter_thread = threading.Thread(target=start_tkinter_mainloop)

# Start the tkinter thread
tkinter_thread.start()

# Now, you can call external_update_window() from your external code
 """

import tkinter as tk
from PIL import Image, ImageTk
from googletrans import Translator
import threading

# Initialize the tkinter window
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("800x600")  # Set window size

# Create a main title label
title_label = tk.Label(root, text="Real Time Translation of Indian Sign Language to Kannada Text",
                       font=("BebasNeue", 20, "bold"), bg="#f0f0f0")
title_label.pack()

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display text with borders and bigger font size
text_label = tk.Label(root, font=("Helvetica", 20))  #, borderwidth=2, relief="solid", bg="white", width=0, height=2
text_label.pack()

translator = Translator()

# Function to translate English text to Kannada
def translate_english_to_kannada(text):
    try:
        translated = translator.translate(text, src='en', dest='kn').text
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation error"

prev = "Bois"
kan = "prev kan"

# Function to update the window with a new frame and text
def update_window(frame, eng_text):
    global prev
    global kan
    # Create a PhotoImage object directly from the frame
    frame_img = ImageTk.PhotoImage(Image.fromarray(frame))

    # Update the image label with the new frame
    image_label.config(image=frame_img)
    image_label.image = frame_img

    # Translate English text to Kannada
    if prev != eng_text:
        kannada_text = translate_english_to_kannada(eng_text)
        prev = eng_text
        kan = kannada_text
    else:
        kannada_text = kan

    # Update the text label with the translated text, with English and Kannada text one below the other
    text_label.config(text=f"English Text: {eng_text}\nKannada Text: {kannada_text}")

    root.update()  # Update the tkinter window

# Function to start the tkinter main loop in a separate thread
def start_tkinter_mainloop():
    root.mainloop()

# Create a thread to run the tkinter main loop
tkinter_thread = threading.Thread(target=start_tkinter_mainloop)

# Start the tkinter thread
tkinter_thread.start()





# Imports
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sv_ttk
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
import warnings
import time
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

warnings.simplefilter(action='ignore', category=FutureWarning)


# Setup the App Structure and Frames
class MainApp(tk.Tk):
    # init function for MainApp
    def __init__(self, *args, **kwargs):
        # init function for Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # Create a container
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        # initializes frames array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (
                HomePage, RealTimeRecognition, PhotoRecognition, VideoRecognition):
            frame = F(container, self)

            # initializing frame of that object from each page planned
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        # Show the home page
        self.show_frame(HomePage)

    # to display the current frame passed as parameter to switch to desired frame of program
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.event_generate("<<ShowFrame>>")


class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        # Setup window dimensions
        window_width = 720
        window_height = 480

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
        body_frame.anchor('center')

        # Load images
        self.rtrBtn = tk.PhotoImage(file="res/img/realtime.png").subsample(8, 8)
        self.prBtn = tk.PhotoImage(file="res/img/photo.png").subsample(8, 8)
        self.vrBtn = tk.PhotoImage(file="res/img/video.png").subsample(8, 8)

        # Real-Time Recognition
        # https://www.flaticon.com/free-icons/time-management created by Abdul-Aziz - Flaticon
        rtrBtn = ttk.Button(body_frame, text="Realtime", image=self.rtrBtn, width=50)
        rtrBtn.pack(ipadx=20, padx=10, pady=10)

        # Photo Recognition
        # https://www.flaticon.com/free-icons/picture - created by Pixel perfect - Flaticon
        prBtn = ttk.Button(body_frame, image=self.prBtn)
        prBtn.pack(ipadx=20, padx=10, pady=10)

        # Video Recognition
        # https://www.flaticon.com/free-icons/video created by Freepik - Flaticon
        vrBtn = ttk.Button(body_frame, image=self.vrBtn)
        vrBtn.pack(ipadx=20, padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

class RealTimeRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)


class PhotoRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)


class VideoRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)


# Main Function
if __name__ == "__main__":
    print('Launching ASL_AI...')

    # Setup MainApp
    app = MainApp()
    app.title("ASL AI Recognition Software")

    # Setup window dimensions
    window_width = 720
    window_height = 480

    # Set the theme of program
    sv_ttk.set_theme("dark")
    print('Theme Set.')
    # Get screen dimensions
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()

    # Find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # Configure Window
    app.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    app.resizable(width=False, height=False)  # Prevent Resizing

    # Run the app
    print('ASL_AI is now running...')
    app.mainloop()

    # On-Exit
    print('ASL_AI has stopped running.')

# Imports
import threading

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sv_ttk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import tensorflow as tf
import warnings
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from PIL import Image, ImageTk

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


class ASLRecognition:
    def __init__(self):
        # Attempt to load the model
        print('Loading ASL Recognition Model...')
        try:
            self.model = load_model('model.h5')
            print('ASL Recognition Model Loaded.')
        except FileNotFoundError:
            # Create model if model is not found
            # Load test and training data
            train_path = 'res/train'
            test_path = 'res/test'

            train_batches = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
                directory=train_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)
            test_batches = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
                directory=test_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)

            # Design the CNN
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
            model.add(MaxPool2D(pool_size=(2, 2), strides=2))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=2))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=2))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.3))
            model.add(Dense(26, activation="softmax"))

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                          metrics=['accuracy'])
            model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

            history = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop],
                                validation_data=test_batches)

            # goto next batch of imgs
            imgs, labels = next(test_batches)

            # Evaluate the model
            scores = model.evaluate(imgs, labels, verbose=0)
            print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

            # Save the trained model
            model.save('model.h5')

            # Plot the model performance over epochs
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(10)

            plt.figure(figsize=(15, 15))
            plt.subplot(2, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

            self.model = model

            print('ASL Recognition Model created and saved.')


class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        # Setup window dimensions
        window_width = 640
        window_height = 480

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=10, fill="x", expand=True)
        body_frame.anchor('center')

        # Title Text
        titleLabel = ttk.Label(body_frame, text='ASLAR', font=("Arial", 25))
        titleLabel.pack(padx=10, pady=10)

        # Load images
        self.rtrBtn = tk.PhotoImage(file="res/img/realtime.png").subsample(10, 10)
        self.prBtn = tk.PhotoImage(file="res/img/photo.png").subsample(10, 10)
        self.vrBtn = tk.PhotoImage(file="res/img/video.png").subsample(10, 10)

        # https://www.flaticon.com/free-icons/time-management created by Abdul-Aziz - Flaticon
        rtrBtn = ttk.Button(body_frame, text="Realtime", image=self.rtrBtn,
                            command=lambda: controller.show_frame(RealTimeRecognition))
        rtrBtn.pack(ipadx=20, padx=10, pady=10)

        # Real-Time Recognition Label
        rtrLabel = ttk.Label(body_frame, text='Real Time ASL Recognition')
        rtrLabel.pack(padx=10, pady=10)

        # Photo Recognition
        # https://www.flaticon.com/free-icons/picture - created by Pixel perfect - Flaticon
        prBtn = ttk.Button(body_frame, image=self.prBtn, command=lambda: controller.show_frame(PhotoRecognition))
        prBtn.pack(ipadx=20, padx=10, pady=10)

        # Photo  Recognition Label
        prBtn = ttk.Label(body_frame, text='Photo ASL Recognition')
        prBtn.pack(padx=10, pady=10)

        # Video Recognition
        # https://www.flaticon.com/free-icons/video created by Freepik - Flaticon
        vrBtn = ttk.Button(body_frame, image=self.vrBtn, command=lambda: controller.show_frame(VideoRecognition))
        vrBtn.pack(ipadx=20, padx=10, pady=10)

        # Photo  Recognition Label
        prBtn = ttk.Label(body_frame, text='Video ASL Recognition')
        prBtn.pack(padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")


class RealTimeRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        print('Loading CV...')
        self.controller = controller

        # Bind visible frame event
        self.bind("<<ShowFrame>>", self.startThread)

        # Create label to capture video frames and display them
        self.video_label = ttk.Label(self)
        self.video_label.place(x=0, y=0, )

        # Place Home Screen Buttom
        self.homeImg = tk.PhotoImage(file='res/img/home.png').subsample(15, 15)
        homeBtn = ttk.Button(self.video_label, image=self.homeImg, width=10, command=self.goHome)
        homeBtn.place(x=640 - 70, y=10)

        # Setup video capture
        self.running = False
        self.background = None
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Setup other vars
        self.num_frames = 0
        self.accumulated_weight = 0.5
        self.count = 0

        # Setup Word Dict
        self.word_dict = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                          'del', 'space']

        # Show video feed
        self.video_label.after(20, self.show_frames)

        self.count = self.count + 1
        print(f'Count: {self.count}')
        if self.count > 1:
            self.run()

    def startThread(self, args):
        # Setup thread
        cvThread = threading.Thread(target=self.run)
        try:
            cvThread.start()
            print('cvThread has started.')
        except RuntimeError:
            print('cvThread is already running.')
            self.run()

    def goHome(self):
        print('Reached goHome().')
        # Reset background and frames
        self.running = False
        self.background = None
        self.num_frames = 0

        # Go to home page
        self.controller.show_frame(HomePage)

    def run(self):
        print('Reached RealTimeRecognition.run()')
        self.running = True
        self.show_frames()

    # Calculate background average weights
    def cal_accum_avg(self, frame, accumulated_weight):
        # Check if there is a background
        if self.background is None:
            self.background = frame.copy().astype("float")
            return None

        cv2.accumulateWeighted(frame, self.background, accumulated_weight)

    # Find Hand in frame and segment it
    def segment_hand(self, frame, threshold=25):
        diff = cv2.absdiff(self.background.astype("uint8"), frame)

        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Extract external contours
        contours, hierarchy = cv2.findContours(
            thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(contours) == 0:
            return None
        else:
            hand_segment_max_cont = max(contours, key=cv2.contourArea)

            return thresholded, hand_segment_max_cont

    # Function to display frames of the video
    def show_frames(self):
        print('Reached show_frames()')
        while self.running:
            print('ASLR Running...')
            ret, frame = self.cap.read()

            # flip the frame to prevent inverted image capture
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            # Setup point of interest in frame
            recognition_box = frame[100:300, 150:350]  # Top:bottom, right:left

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(recognition_box, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)  # Smooth the image

            # Get the background
            if self.num_frames < 120:
                # Scan background for later comparison
                self.cal_accum_avg(gray_frame, self.accumulated_weight)
                cv2.putText(frame_copy, "Scanning Background... Please wait.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # Segment the hand in the recognition box
                hand = self.segment_hand(gray_frame)

                # Check if hand is present in recognition box
                if hand is not None:
                    thresh_holded, hand_segment = hand

                    # Draw contours for the segmented hand
                    cv2.drawContours(frame_copy, [hand_segment + (150, 100)], -1, (255, 0, 0))

                    # Rehape/resize frame
                    thresh_holded = cv2.resize(thresh_holded, (64, 64))
                    thresh_holded = cv2.cvtColor(thresh_holded, cv2.COLOR_GRAY2RGB)
                    thresh_holded = np.reshape(thresh_holded,
                                               (1, thresh_holded.shape[0], thresh_holded.shape[1], 3))
                    # Make prediction
                    pred = asl.model.predict(thresh_holded)
                    cv2.putText(frame_copy, f'Prediction: {self.word_dict[np.argmax(pred)]}',
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Print prediction for debugging
                    print(np.argmax(pred))

            # Draw the recognition box
            cv2.rectangle(frame_copy, (350, 100), (150, 300), (255, 128, 0), 3)

            # incrementing the number of frames for tracking
            self.num_frames += 1

            # Display the frame with segmented hand
            rgbIm = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgbIm)

            imgtk = ImageTk.PhotoImage(image=im)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            # cv2.imshow("ASL Recognition", frame_copy)


class PhotoRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.img_file = None

        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=10, fill="x", expand=True)
        body_frame.anchor('center')

        img_frame = ttk.Frame(body_frame, width=300)
        img_frame.pack()

        img_pred_label = ttk.Label(body_frame, text='Waiting on image...')
        img_pred_label.pack()

        file_select_btn = ttk.Button(body_frame, text='Select an Image', command=self.select_file)
        file_select_btn.pack()

        classification_btn = ttk.Button(body_frame, text='Predict ASL', command=self.classify_img)
        classification_btn.pack()

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

    def select_file(self):
        # File chooser
        filetypes = (
            ('PNG', '*.png'),
            ('JPG', '*.jpg'),
            ('JPEG', '*.jpeg')
        )

        # Load file from the user
        self.img_file = filedialog.askopenfilename(title='Select an Image of ASL',
                                                   initialdir='/',
                                                   filetypes=filetypes)

    def classify_img(self):
        classification = asl.model.predict


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
    window_width = 640
    window_height = 480

    # Set the theme of program
    sv_ttk.set_theme("light")
    print('Theme Set.')

    # Load the ASL Recognition Model
    asl = ASLRecognition()

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

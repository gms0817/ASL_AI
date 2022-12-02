import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


# Defining CreateWidgets() function to create necessary tkinter widgets
def createwidgets():
    root.feedlabel = Label(root, bg="#d4a373", fg="white", text="Camera Input", font="Times 40 bold")
    root.feedlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    root.cameraLabel = Label(root, bg="#ccd5ae", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)

    root.stopCameraButton = Button(root, text="Stop Camera", command=stopCamera, bg="#faedcd", font=('Times', 25),
                         width=13)
    root.stopCameraButton.grid(row=3, column=1, padx=10, pady=10,columnspan=2)

    # Calling ShowFeed() function
    ShowFeed()


# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()

    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)

        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)

        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image=videoImg)

        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)

        # Keeping a reference
        root.cameraLabel.imgtk = imgtk

        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image='')


# Defining StopCAM() to stop WEBCAM Preview
def stopCamera():
    # Stopping the camera using release() method of cv2.VideoCapture()
    root.cap.release()

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="START CAMERA", command=StartCAM)


def StartCAM():
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)

    # Setting width and height
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="STOP CAMERA", command=stopCamera)

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFeed()


# Creating object of tk class
root = tk.Tk()

# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)

# Setting width and height
width, height = 640, 480
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Setting the title, window size, background color and disabling the resizing property
root.title("AIForAll Group 4: ASL Training")
root.geometry("1340x700")
root.resizable(True, True)

createwidgets()
root.mainloop()

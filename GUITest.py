from tkinter import *
import cv2

root = Tk()

def openCamera():
    cap = cv2.VideoCapture(0)
    while TRUE:
        ret, frame = cap.read()
        #width = int(cap.get(3))
        #height = int(cap.get(4))
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()

cameraBtn = Button(root, text="Click to Open Camera", command=openCamera)
exitBtn = Button(root, text="Exit Camera")

cameraBtn.pack()
exitBtn.pack()
root.mainloop()

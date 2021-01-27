import cv2
import string

class Camera():
    """Class provides tools to open camera stream and run main image recognition loop."""
    
    cameraId = 0
    width = 0
    height = 0
    windowName = ""
    cap = None
    loopHandle = None
    arguments = None
    key = ""
    firstFrame = None
    firstFrameSet = False
    onceSuccess = False
    lastResult = "" 


    def openStream(self):
        self.cap = cv2.VideoCapture(self.cameraId)
        if not self.cap.isOpened():
            print(f"ERROR: Could not open video device - id: {self.cameraId}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)   
        return True 

    def putText(self, image, text, position = (10,25)):
        font = cv2.FONT_HERSHEY_SIMPLEX   
        fontScale = 0.5
        color = (0, 255, 0) 
        thickness = 1
        cv2.putText(image, text, position, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        return image

    def initCameraLoop(self, show=True):

        if self.openStream():
            if show:
                cv2.namedWindow(self.windowName)
                cv2.namedWindow("proc image")

                # window position is adjusted for screen with 2560x1440 resolution 
                cv2.moveWindow("proc image", 110,430); 

            # CAMERA LOOP
            while(True):

                ret, frame = self.cap.read()

                if not ret:
                    print("ERROR: Camera stream - no image returned")
                    break

                # pack arguments and run hand recognition from handle 
                self.arguments[0] = frame
                self.arguments[1] = self.key
                res, result = self.loopHandle(*self.arguments)

                # interface behaviour                
                if result is not None: 
                    self.lastResult = result
                    self.onceSuccess = True
                    #clear on screen text if 'c' is pressed
                    if result[0] == "clear":
                        self.onceSuccess = False

                if show:
                    if res is not None:
                        cv2.imshow("proc image", res)                
                    
                    if self.onceSuccess:
                        self.putText(frame,f"Detector result: {self.lastResult[0]}")                        
                    else:
                        self.putText(frame,"ASL detector - press 'r' to detect")

                    cv2.imshow(self.windowName, frame)

                self.key = cv2.waitKey(1)
                if self.key == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    # only for development - test how background subtraction works in case segmentation fails 
    def backgroundSubtraction(self, frame):
        if self.firstFrameSet == False:
            first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            first_gray = cv2.GaussianBlur(first_gray, (7, 7), 0)
            self.firstFrame = first_gray
            self.firstFrameSet = True
            return self.firstFrame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            difference = cv2.absdiff(gray, self.firstFrame)
            thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]
     
            thresh = cv2.dilate(thresh, None, iterations=2)
            return thresh        


    def __init__(self, method = None, args = None, cameraId = 0, width=320, height=240, windowName = "camera stream"):
        self.cameraId = cameraId
        self.width = width
        self.height = height
        self.windowName = windowName
        self.loopHandle = method
        self.arguments = args

# camera sandbox - in case of camera stream problems - debug it from here
# def main():

#     def cameraLoopMethod(frame, key):
#         grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
#         if key == ord('r'):
#             print("Message from camera loop method")
#         return grayFrame 

#     camera = Camera(method=cameraLoopMethod, args=[None, None])
#     camera.initCameraLoop()

# if __name__ == "__main__":
#     main()

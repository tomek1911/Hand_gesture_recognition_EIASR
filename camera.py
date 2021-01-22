
import cv2

class Camera():
    
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

    def openStream(self):
        self.cap = cv2.VideoCapture(self.cameraId)
        if not self.cap.isOpened():
            print("Could not open video device")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)   
        return True 

    #atrapa

    def initCameraLoop(self, show=True):

        if self.openStream():
            if show:
                cv2.namedWindow(self.windowName)
                # cv2.namedWindow("subtracted background")
                cv2.namedWindow("proc image")

            while(True):

                ret, frame = self.cap.read()

                if not ret:
                    break

                self.arguments[0] = frame
                self.arguments[1] = self.key
                res = self.loopHandle(*self.arguments)
                # sub = self.backgroundSubtraction(frame)

                if show:
                    cv2.imshow(self.windowName, frame)
                    # cv2.imshow("subtracted background",sub)
                    if res is not None:
                        cv2.imshow("proc image", res)

                self.key = cv2.waitKey(1)
                if self.key == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


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
            #+cv2.THRESH_OTSU
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


def main():

    #Atrapa funkcji przetwarzania obrazu
    #obsluga wejscia i wyjscia

    def cameraLoopMethod(frame, key):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        if key == ord('r'):
            print("Run forest, run!!!")
        return grayFrame 

    camera = Camera(method=cameraLoopMethod, args=[None, None])
    camera.initCameraLoop()


if __name__ == "__main__":
    main()


import cv2

class Camera():

    
    cameraId = 0
    width = 0
    height = 0
    windowName = ""
    cap = None
    loopHandle = None
    arguments = None

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
                
            while(True):

                ret, frame = self.cap.read()

                #do something here

                res = self.loopHandle(*self.arguments)    

                if show:  
                    cv2.imshow(self.windowName, frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()



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

    def cameraLoopMethod(message1, messeage2):
        print(message1+messeage2)
        return 0

    camera = Camera(method=cameraLoopMethod, args=["Hello", "from camera loop"])
    camera.initCameraLoop()


if __name__ == "__main__":
    main()

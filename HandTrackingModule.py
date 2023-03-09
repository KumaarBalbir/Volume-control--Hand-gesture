import cv2
import mediapipe as mp

# to check the frame rate
import time
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 21 landmark points

    def findHands(self,img,draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the frame
    #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    #Draw dots and connect them
                    self.mpDraw.draw_landmarks(img,handLms,
                                                self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,handNo=0,draw=True):

        # initialise a list to store all the landmarks position for handNo
        lmList=[]

        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]


            # grab each landmarks (21) and its position
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                # get height , width and channel of image
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id,cx,cy])
                # now you can grab any landmark by its id and do operation on them

                if draw:
                    # drawing a cirle with centre (cx,cy) and radius 15
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return  lmList



def main():
    # tracking the frame rate
    # previous time
    pTime = 0
    # current Time
    cTime = 0

    # get the frame object
    cap = cv2.VideoCapture(0)

    # create an object of handDetector
    detector=handDetector()


    while True:
        success, img = cap.read()

        # pass the original image to detector and it will return image with landmarks
        img= detector.findHands(img)

        # print the list of positions of a particular landmark
        lmList=detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

















if __name__=="__main__":
    main()
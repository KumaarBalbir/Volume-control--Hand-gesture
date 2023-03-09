import cv2
import mediapipe as mp

# to check the frame rate
import time

# get the frame object
cap =cv2.VideoCapture(0)

# Detect the hand
# to get the details about functions arg and its definition, ctr and click by hovering over the defn
mpHands=mp.solutions.hands
# static_image_mode=False, means detecting the hand upon some threshold otherwise it won't be detected
# for this project all default value of Hands() method is fine for us
hands=mpHands.Hands()

# making function to draw handLandmarks
mpDraw = mp.solutions.drawing_utils

# tracking the frame rate
# previous time
pTime=0
# current Time
cTime=0


while True:
    success,img=cap.read()
    # since, Hands() object uses only RGB images
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # process the frame and give the result
    results=hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:

        # loop over each hand Landmarks
        for handLms in results.multi_hand_landmarks:

            # grab each landmarks (21) and its position
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                # get height , width and channel of image
                h,w,c =img.shape
                cx,cy=int(lm.x*w), int (lm.y*h)
                print(id,cx,cy)
                # now you can grab any landmark by its id and do operation on them
                if id==4:
                    # drawing a cirle on id=4 with centre (cx,cy) and radius 15
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            # draw hand landmarks using draw_landmarks function
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)

    cv2.imshow("Image displaying..",img)
    cv2.waitKey(1)

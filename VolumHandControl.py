import cv2
import numpy as np
import time
import  HandTrackingModule as htm
import math

# library for audio management
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
###############################

wCam,hCam= 600, 480

###############################

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

# prev time & current time
pTime=0
cTime=0

# create an object of htm class, with 0.7 detection confidence
detector=htm.handDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange= volume.GetVolumeRange()
print(volRange)
minVol=volRange[0]
maxVol=volRange[1]



vol=0
volBar=400
volPer=0


while True:
    success,img=cap.read()

    # call handDetector
    img=detector.findHands(img)
    # get the list of landmarks position
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        # print(lmList[4],lmList[8])
        # store x,y coordinate of landmark 4 & 8
        x1,y1= lmList[4][1],lmList[4][2]
        x2,y2= lmList[8][1],lmList[8][2]

        # get the centre of the landmark 4 & 8

        cx,cy=(x1+x2)//2 , (y1+y2)//2

        # draw circle for landmark 4, & 8

        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.circle(img,(cx,cy),15,(255,255,0),cv2.FILLED)

        # get the distance between lm 4 & 8
        length=math.hypot(x2-x1,y2-y1)
        # print(length)
        # Hand Range 50-200
        # vol range -60-0
        vol=np.interp(length,[50,200],[minVol,maxVol])
        # volume for bar, when vol is minimum i.e. 50 the rect filled will be at height 400
        volBar = np.interp(length, [50, 200], [400, 150])
        # show vol as percentage
        volPer = np.interp(length, [50, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)


        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img,(50,150),(85,400),(255,0,0),2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0),cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    cv2.imshow("IMage being captured..",img)
    cv2.waitKey(1)
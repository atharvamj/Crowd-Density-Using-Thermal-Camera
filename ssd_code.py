import cv2

import matplotlib.pyplot as plt


# ### Installing SSD mobilenet with coco label dataset


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


model = cv2.dnn_DetectionModel(frozen_model, config_file)


classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


print(classLabels)


print(len(classLabels))


model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)  # 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# # Read image


img = cv2.imread('06.png')


plt.imshow(img)  # This image is in BGR format as standard


# Convert in RGB using cv2.COLOR_BGR2RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


ClassIndex, confidece, bbox = model.detect(img, confThreshold=0.5)


# pip install --upgrade opencv-python        ### For upgrading opencv to latest version, only required to run once


print(ClassIndex)


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40),
                font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


### Shows the output in separate window ###
ims = cv2.resize(img, (1920, 1080))
cv2.imshow("output", ims)
cv2.waitKey(0)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# Video


cap = cv2.VideoCapture('IMG_8510.mp4')  # capture/run the video

cap.set(4, 720)  # set resolution as 720x720
cap.set(3, 720)  # set resolution as 720x720

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN


while True:
    ret, frame = cap.read()

    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.55)

    print(ClassIndex)
    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd <= 80):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    #cv2.imshow('Obj Detection', frame)
    ims = cv2.resize(frame, (1920, 1080))
    cv2.imshow("output", ims)
    # cv2.waitKey()
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
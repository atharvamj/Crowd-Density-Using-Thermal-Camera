import cv2

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture('./data/video/test_4.mp4')
i = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False or i > 50:
        break

    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite('Frame' + str(i) + '.jpg', frame)
    i += 1
    print(i)
cap.release()
cv2.destroyAllWindows()
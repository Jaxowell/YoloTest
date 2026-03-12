import cv2
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
cv2.imshow("TEST", frame)
cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
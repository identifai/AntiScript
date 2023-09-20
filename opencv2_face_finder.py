import cv2 as cv
im = cv.imread(filepath)
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cascade = cv.data + "haarcascade_frontalface_default.xml"
classifier = cv.CascadeClassifier(cascade)

faces = classifier.detectMultiScale(
    im_gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(50, 50)
)

print("Found {} face candidates".format(len(faces))

# Draw rectangles over faces
for (x, y, w, h) in faces:
    cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv.imshow("Result" ,im)
cv.waitKey(0)
cv.destroyAllWindows()

import cv2
import face_recognition

# Load reference images
imgSheku = face_recognition.load_image_file('ImagesBasic/Sheku k.jpg')
imgSheku = cv2.cvtColor(imgSheku, cv2.COLOR_BGR2RGB)
imgVinayak = face_recognition.load_image_file('ImagesBasic/Vinayak c.jpg')
imgVinayak = cv2.cvtColor(imgVinayak, cv2.COLOR_BGR2RGB)
imgAkash = face_recognition.load_image_file('ImagesBasic/Akash b.jpg')
imgAkash = cv2.cvtColor(imgAkash, cv2.COLOR_BGR2RGB)
imgSumit = face_recognition.load_image_file('ImagesBasic/Sumit d.jpg')
imgSumit = cv2.cvtColor(imgSumit, cv2.COLOR_BGR2RGB)
imgDarshan = face_recognition.load_image_file('ImagesBasic/Darshan g.jpg')
imgDarshan = cv2.cvtColor(imgDarshan, cv2.COLOR_BGR2RGB)
imgBrunda = face_recognition.load_image_file('ImagesBasic/Brunda G.jpg')
imgBrunda = cv2.cvtColor(imgDarshan, cv2.COLOR_BGR2RGB)
imgParvati = face_recognition.load_image_file('ImagesBasic/Parvati H.jpg')
imgParvati = cv2.cvtColor(imgDarshan, cv2.COLOR_BGR2RGB)

# Load test image
imgTest = face_recognition.load_image_file("C:\\Users\\katra\\PycharmProjects\\ANC\\imagesAttendance\\Sheku k.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Face encoding for reference images
faceLocSheku = face_recognition.face_locations(imgSheku)[0]
encodeSheku = face_recognition.face_encodings(imgSheku)[0]
cv2.rectangle(imgSheku, (faceLocSheku[3], faceLocSheku[0]), (faceLocSheku[1], faceLocSheku[2]), (255, 0, 255), 2)

# Face encoding for test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Compare faces
results = face_recognition.compare_faces([encodeSheku], encodeTest)
faceDis = face_recognition.face_distance([encodeSheku], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Display images
cv2.imshow('Sheku k', imgSheku)
cv2.imshow('Sheku test', imgTest)
cv2.imshow('Vinayak c', imgVinayak)
cv2.imshow('Vinayak test', imgTest)
cv2.imshow('Akash b', imgAkash)
cv2.imshow('Akash test', imgTest)
cv2.imshow('Sumit d', imgSumit)
cv2.imshow('Sumit test', imgTest)
cv2.imshow('Darshan g', imgDarshan)
cv2.imshow('Darshan test', imgTest)
cv2.imshow('Brunda G', imgDarshan)
cv2.imshow('Brunda test', imgTest)
cv2.imshow('Parvati H', imgDarshan)
cv2.imshow('Parvati test', imgTest)

cv2.waitKey(0)

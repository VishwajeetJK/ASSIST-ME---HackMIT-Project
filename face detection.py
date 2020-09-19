import face_recognition

picture_of_me = face_recognition.load_image_file("/Users/sahaaj/Downloads/H4C/img/Unknown.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("/Users/sahaaj/Downloads/H4C/img/Unknown1.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces(my_face_encoding, unknown_face_encoding)

if results == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")

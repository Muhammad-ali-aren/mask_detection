
import torch
import os
from torchvision import transforms as T
import cv2 as cv
from main import model_path
from faceNet import FaceNet

model = FaceNet(in_channels=3,hidden_size=16,output_shape=1)
model.load_state_dict(torch.load(model_path))
model.eval()
print('Model loaded successfully...')

cascade_path = r"C:\Users\M-Ali\Downloads\New Masks Dataset\haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cascade_path)

def detect_face(img):
    face_img = img.copy()
    face_rec =  face_cascade.detectMultiScale(face_img, scaleFactor=1.2,minNeighbors=2)
    print(face_rec)
    if len(face_rec) == 0 :
        return None
    for (x,y,w,h) in face_rec:
        return x,y,w,h

data_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(size=(200,200)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    
])

def preprocess_img(img):
    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    image = data_transform(image)
    image = image.unsqueeze(0)
    return image

def result_det(preds):
    response = (preds > 0.5).int().item()
    output = None
    color = None
    if response == 0:
        output = 'Mask'
        color = (0,255,0)
    elif response == 1:
        output = 'No Mask'
        color = (0,0,255)
    return output,color

def real_time_detection(video_path=None,camera=None):
    cap = cv.VideoCapture(camera)

    while True:
        ret, frame = cap.read()
        face_coords = detect_face(frame)
        if face_coords is not None:
            x,y,w,h = face_coords
            crop_frame = frame[y:y+h,x:x+w]
            image = preprocess_img(crop_frame)
            with torch.inference_mode():
                pred = model(image)
            output, color = result_det(pred)
            cv.rectangle(frame,(x,y),(x+w,y+h),color,5)
            cv.putText(frame,output,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.9,color,2)
            cv.imshow('Mask Detection',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv.destroyWindow('Mask Detection')


real_time_detection(camera=0)






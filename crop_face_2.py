##################################################################
# This script crop faces from a folder contains many human figures
##################################################################
# python crop_face_2.py train/cage train/cage_face/
import sys
import dlib
import cv2
import os

Images_Folder = sys.argv[1]
OutFace_Folder = sys.argv[2]
Images_Path = os.path.join(os.path.realpath('.'), Images_Folder)
pictures = os.listdir(Images_Path)

dnnFaceDetector = dlib.cnn_face_detection_model_v1("./dlib/mmod_human_face_detector.dat")
print(pictures)

def rotate(img):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

for f in pictures:
    img = cv2.imread(os.path.join(Images_Path,f), cv2.IMREAD_COLOR)
    dets = dnnFaceDetector(img, 0)
    print("Number of faces detected: {}".format(len(dets)))
    for idx, face in enumerate(dets):
        left = face.rect.left()
        top = face.rect.top()
        right = face.rect.right()
        bot = face.rect.bottom()
        crop_img = img[top:bot, left:right]
        cv2.imwrite(OutFace_Folder+f[:-4]+"_face.jpg", crop_img)



import cv2
import math
import numpy as np
from PIL import Image
import scipy.misc
import os

face_detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascades/haarcascade_eye1.xml")

skewed = "skewed"
fixed = "fixed"
input = "data"
output = "output"
finaloutput = "finaloutput"

if not os.path.exists(skewed):
    os.makedirs(os.path.join(skewed))

if not os.path.exists(fixed):
    os.makedirs(os.path.join(fixed))


if not os.path.exists(output):
    os.makedirs(os.path.join(output))

if not os.path.exists(finaloutput):
    os.makedirs(os.path.join(finaloutput))

def fix(num):
    ret = np.zeros([num, 6])
    #ret: angle, skewed-startX, skewed-startY, width, height, direction
    for i in np.arange(num):
        stri = str(i+1)
        if (i+1 < 10):
            stri = "0000" + stri
        elif (i+1 < 100):
            stri = "000" + stri
        elif (i+1 < 1000):
            stri = "00" + stri
        elif (i+1 < 10000):
            stri = "0" + stri
        img = cv2.imread(input + "/" + stri + ".jpg")

        img_raw = img.copy()
        img_slay = img_raw.copy()

        faces = face_detector.detectMultiScale(img, 1.3, 5)
        face_x, face_y, face_w, face_h = faces[0]

        img = img[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        eyes = eye_detector.detectMultiScale(img_gray)

        index = 0
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            if index == 0:
                eye_1 = (eye_x, eye_y, eye_w, eye_h)
            elif index == 1:
                eye_2 = (eye_x, eye_y, eye_w, eye_h)

            cv2.rectangle(img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)
            index = index + 1

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0];
        left_eye_y = left_eye_center[1]

        right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        right_eye_x = right_eye_center[0];
        right_eye_y = right_eye_center[1]

        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 2)
        cv2.line(img, right_eye_center, left_eye_center, (67, 67, 67), 2)

        if left_eye_y < right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            print("rotate to inverse clock direction")

        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)

        cv2.line(img, right_eye_center, left_eye_center, (67, 67, 67), 2)
        cv2.line(img, left_eye_center, point_3rd, (67, 67, 67), 2)
        cv2.line(img, right_eye_center, point_3rd, (67, 67, 67), 2)

        def euclidean_distance(a, b):
            x1 = a[0];
            y1 = a[1]
            x2 = b[0];
            y2 = b[1]
            return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        print("cos(a) = ", cos_a)

        angle = np.arccos(cos_a)
        print("angle: ", angle, " in radian")

        angle = (angle * 180) / math.pi
        print("angle: ", angle, " in degree")

        if direction == -1:
            angle = 90 - angle

        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * -1 * angle))

        facess = face_detector.detectMultiScale(new_img, 1.3, 5)
        face_xs, face_ys, face_ws, face_hs = facess[0]

        cropped = new_img[int(face_ys):int(face_ys + face_hs), int(face_xs):int(face_xs + face_ws)]



        cv2.imwrite(skewed+"/"+stri+".jpg", new_img)

        cropped = cv2.resize(cropped, (182, 182))
        cv2.imwrite(fixed + "/" + stri + ".jpg", cropped)

        ret[i, 0] = angle
        ret[i, 1] = face_xs
        ret[i, 2] = face_ys
        ret[i, 3] = face_ws
        ret[i, 4] = face_hs
        ret[i, 5] = direction

    return ret

def reattach(arr):
    for i in np.arange(np.shape(arr)[0]):
        stri = str(i+1)
        if (i+1 < 10):
            stri = "0000" + stri
        elif (i+1 < 100):
            stri = "000" + stri
        elif (i+1 < 1000):
            stri = "00" + stri
        elif (i+1 < 10000):
            stri = "0" + stri

        angle = arr[i, 0]
        face_xs = int(arr[i, 1])
        face_ys = int(arr[i, 2])
        face_ws = int(arr[i, 3])
        face_hs = int(arr[i, 4])
        direction = int(arr[i, 5])

        origimg = cv2.imread(input + "/" + stri + ".jpg")
        skewedimg = cv2.imread(skewed+"/"+stri+".jpg")
        fixedimg = cv2.imread(output + "/" + stri + ".jpg")
        # print(face_ws)
        # print(face_hs)
        # print(np.shape(fixedimg))
        fixedimg = fixedimg[10:172, 10:172].copy()
        changesssw = int(face_ws - 2*face_ws*10/182)
        changesssh = int(face_hs - 2*face_hs * 10 / 182)
        fixedimg = cv2.resize(fixedimg, (changesssw, changesssh))

        face_xs += int(face_ws*10/182)
        face_ys += int(face_hs*10 / 182)

        skewedimg[face_ys:face_ys + np.shape(fixedimg)[0], face_xs:face_xs + np.shape(fixedimg)[1]] = fixedimg

        rotated = Image.fromarray(skewedimg)
        rotated = np.array(rotated.rotate(direction * 1 * angle))
        naur = np.where((rotated == [0, 0, 0]).all(axis=2), -1, 100)

        hehe = np.zeros([np.shape(skewedimg)[0], np.shape(skewedimg)[1], 3])
        for i in np.arange(np.shape(rotated)[0]):
            for j in np.arange(np.shape(rotated)[1]):
                if (naur[i, j] < 0):
                    for k in np.arange(3):
                        hehe[i, j, k] = origimg[i, j, k]
                        # print("img_slay ", img_slay[i, j, k])
                    # print("slay")
                else:
                    for k in np.arange(3):
                        hehe[i, j, k] = rotated[i, j, k]


        toret = np.uint8(hehe)
        cv2.imwrite(finaloutput+"/"+stri+".jpg", toret)


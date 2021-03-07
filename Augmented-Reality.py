import cv2
import numpy as np

input_image = cv2.imread('Maha.jpeg')
input_image = cv2.resize(input_image, (300,300))
aug_image = cv2.imread('green.jpeg')
aug_image = cv2.resize(aug_image, (300,300))

w, h, c = input_image.shape

cap = cv2.VideoCapture(0)

#features
detector1 = cv2.ORB_create(1000)
kp = detector1.detect(input_image, None)
descriptor1 = cv2.xfeatures2d.BEBLID_create(0.75)
kp, des = descriptor1.compute(input_image, kp)

#kp, des = orb.detectAndCompute(input_image, None)
#input_image = cv2.drawKeypoints(input_image, kp, input_image)

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

while True:
    _, grayframe = cap.read()
    grayframe = cv2.resize(grayframe, (600,600))
    grayframe_bw = cv2.cvtColor(grayframe, cv2.COLOR_BGR2GRAY)

    detector2 = cv2.ORB_create(1000)
    kp_gframe = detector2.detect(grayframe_bw, None)
    descriptor2 = cv2.xfeatures2d.BEBLID_create(0.75)
    kp_gframe, des_gframe = descriptor2.compute(grayframe_bw, kp_gframe)



    matches = bf.match(des, des_gframe)


    matches = sorted(matches, key = lambda x: x.distance)


    img3 = cv2.drawMatches(input_image, kp, grayframe, kp_gframe, matches[:5], None, flags= 2)

    # Homography

    if len(matches) > 215:
        img_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) #query
        frame_pts = np.float32([kp_gframe[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) #train

        matrix, mask = cv2.findHomography(img_pts, frame_pts, cv2.RANSAC, 5.0)
        print(len(matches))
        print(matrix)
        matches_mask = mask.ravel().tolist()

        # perspective transformation
        w, h , c = input_image.shape

        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(grayframe, [np.int32(dst)], True, (255, 0, 255), 3)

        m_aug = cv2.warpPerspective(aug_image, matrix, (600,600))
        frame_aug = cv2.fillConvexPoly(grayframe, dst.astype(int), (0,0,0))

        final = frame_aug+m_aug

    #cv2.imshow('input', input_image)
    #cv2.imshow('frame', grayframe)
        #cv2.imshow('match', img3)
    #cv2.imshow('homography', homography)
    #cv2.imshow('pers', imgWarp)
    #cv2.imshow('masknew', imgAug)
    #cv2.imshow('aug', aug_image)
    #cv2.imshow('img2', img2)

        #if img2:
        cv2.imshow('final out', final)

    else:
        cv2.imshow('final out', grayframe)

    k = cv2.waitKey(1)
    if k == 27:
        break



cap.release()
cv2.destroyAllWindows()

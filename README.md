Augmented-Reality-using-OpenCV-Python

In this code, I first detected the key points using and descriptors using ORB detector and BEBLID detector from the input image. Then I detected the key points and descriptors from the image shown in the web cam frame. Here I used Brute Force Matcher to match to the key points of the both images for feature matching. Once we get the features we segregate the points using the concept of Hamming distance. After applying Homography, in the end we augmented another image onto the input image. 

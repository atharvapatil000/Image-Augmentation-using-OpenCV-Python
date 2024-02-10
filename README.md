Image-Augmentation-using-OpenCV-Python

This code implements feature matching and image augmentation using various techniques. Here's a breakdown of the process:

1. Key points and descriptors are detected using both ORB detector and BEBLID detector on the input image.
2. Key points and descriptors are then detected from the image shown in the webcam frame.
3. The Brute Force Matcher is employed to match the key points of both images for feature matching.
4. The Hamming distance concept is utilized to segregate the good points from all the matched points.
5. Homography is applied to align the images properly.
6. Finally, another image is augmented onto the input image.

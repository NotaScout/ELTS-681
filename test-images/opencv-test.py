import cv2 as cv
import numpy as np


# image = cv.imread("test-images/testpic.jpg")

# cv.imshow("image", image)

# cv.waitKey(0)



# read an image file
img = cv.imread("test-images/testpic.jpg")
print(img.shape)
# color to gray
gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gimg.shape)
# display the image
# cv.imshow("Original RGB image", img)
# cv.waitKey(0)
# delete window
# cv.destroyAllWindows()
# save an image file
shrink = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

cv.imwrite('test-images/output_testimage.png', shrink)

M2=np.float32([[1,0,40],[0,1,0]])

trans = cv.warpAffine(img, M2, (640, 480))

cv.imwrite('test-images/output_testimage_translate.png', trans)

width = 640
height = 480

imgcenter = (width/2, height/2)

rotM = cv.getRotationMatrix2D(imgcenter, 45, 0.8)

rotimg = cv.warpAffine(img, rotM, (width, height))

cv.imwrite('test-images/output_testimage_rotate.png', rotimg)

pts1=np.float32([[20,10],[40,10],[20,20]])
pts2=np.float32([[20,30],[30,20],[20,40]])
affM = cv.getAffineTransform(pts1,pts2)
affimg = cv.warpAffine(img, affM, (width, height))

cv.imwrite('test-images/output_testimage_affine.png', affimg)
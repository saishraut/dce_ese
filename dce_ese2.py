import numpy as np
import cv2
import random


v_max = 10
mu_min = 4
S = 0.1


transaction_id = random.randint(0,255)
print "transaction id int: "+ str(transaction_id)
transaction_id_str = '{0:08b}'.format(transaction_id)
print "transaction id bits: "+transaction_id_str


img = cv2.imread('coco.jpg')
img_lum = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
cv2.imshow('image',img_lum)
cv2.waitKey(0)
cv2.destroyAllWindows()

channels=cv2.split(img_lum)
y = channels[0]
cb = channels[1]
cr = channels[2]
#print channels[0]
#print img_lum
hpf_mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
img_hpf = cv2.filter2D(y, -1, hpf_mask)
lambda_img = S* abs(img_hpf)
img_res = cv2.cvtColor(img_lum, cv2.COLOR_YCR_CB2BGR)


cv2.imshow('image',img_res)

cv2.waitKey(0)
cv2.destroyAllWindows()



###########################################################################
'''
import numpy as np
import cv2
cap = cv2.VideoCapture('waterfall.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''

'''
cap = cv2.VideoCapture('waterfall.mp4')
if not cap.isOpened(): 
    print "could not open :"
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

while(True):
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

	hpf_mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	img_hpf = cv2.filter2D(img_gaussian, -1, hpf_mask)

	# Display the resulting frame
	frame_size = (500,300)
	gray=cv2.resize(gray,frame_size)

	cv2.imshow('frame_gray',gray)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

'''



'''
def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    # create a CLAHE object
    clahe = cv2.createCLAHE()
    channels[0] = clahe.apply(channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)


'''
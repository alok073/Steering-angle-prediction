import cv2

#provide the path of the video
path1 = r'C:\Users\alok\Downloads\cap1.mp4'
cap = cv2.VideoCapture(path1)
print(cap.get(cv2.CAP_PROP_FPS))  #fps
print(cap.get(cv2.CAP_PROP_POS_MSEC)) #timestamp

current = 0
sec=0

ret=True
while(ret==True):
    ret, frame = cap.read()
    path = r'C:\video_to_frame\frames\frame' +  str(current) + '.jpg'
    print(ret)
    print(path)
    if(ret==True):
        #change the fps
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        cv2.imwrite(path, frame)
    current += 1
    sec = sec + 0.1  #10fps (approx)
    sec = round(sec,2)

cap.release()
cv2.destroyAllWindows()
    

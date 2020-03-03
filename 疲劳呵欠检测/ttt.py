import os
import cv2
path = "pic"
filelist = os.listdir(path)
path_list = [os.path.join(path, file) for file in filelist if not file.startswith("__init__")]
print(filelist)
print(path_list)
for file in path_list:
    img = cv2.imread(file)
    cv2.imshow("img", img )
    cv2.waitKey()
    cv2.destroyAllWindows()
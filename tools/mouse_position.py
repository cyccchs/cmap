import cv2
import os

with open("./HRSC2016/Train/AllImages/image_names.txt", "r") as f:
	img_list = [i.strip() for i in f.readlines()]

def draw(img, img_path):
	root, name = os.path.split(img_path)
	xml = os.path.join(root.replace("AllImages", "Annotations"), name[:-4]+'.xml')
	with open(xml, 'r') as f:
		content = f.read()
		objects = content.split('<HRSC_Object>')
		info = objects.pop(0)
		for obj in objects:
			cls_id = obj[obj.find('<Class_ID>')+10 : obj.find('</Class_ID>')]
			cx = round(eval(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')]))
			cy = round(eval(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')]))
			cv2.putText(img, cls_id[7:9], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

i = 0
img = cv2.imread(img_list[i])
draw(img, img_list[i])

while(1):
	cv2.imshow(os.path.basename(img_list[i]), img)
	key = cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	if key == 13:
		i = i - 1
		img = cv2.imread(img_list[i])
		draw(img, img_list[i])
	if key == 32:
		i = i + 1
		img = cv2.imread(img_list[i])
		draw(img, img_list[i])
	if key == 27:
		break
	


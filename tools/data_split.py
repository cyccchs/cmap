import os
import random
import shutil
src_dir = "./HRSC2016/FullDataSet"
dst_dir = "./train"

with open("./HRSC2016/FullDataSet/AllImages/name.txt", 'r') as f:
    name_list = [i.strip() for i in f.readlines()]
#print(name_list)
clone_list = []
for i in range(273):
    index = round(random.random()*len(name_list))
    if index > len(name_list):
        index = len(name_list)
    clone_list.append(name_list.pop(index).replace(".bmp", ""))
print(clone_list)

for name in clone_list:
    src_img_path = os.path.join(src_dir,"AllImages", name) + ".bmp"
    src_ann_path = os.path.join(src_dir,"Annotations", name) + ".xml"
    dst_img_path = os.path.join(dst_dir,"AllImages", name) + ".bmp"
    dst_ann_path = os.path.join(dst_dir,"Annotations", name) + ".xml"
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_ann_path, dst_ann_path)
    print(src_img_path)
    print(src_img_path)
    print(dst_img_path)
    print(dst_ann_path)


import os
import numpy as np 
import matplotlib.image as img
from sklearn.neural_network import MLPClassifier 
from skimage.feature import canny
path = []
for root, dirs, files in os.walk('Bilal'):
    if "train" in dirs:
        for fname in sorted(dirs):  
            src = os.path.join(root, fname)
            path.append(src)

dir_list = sorted(os.listdir(path[0]))
dir_list.pop(0)
train_images = []
train_images_count = []
for curr_path in dir_list:
    src = os.path.join(path[0],curr_path)
    current_dir = os.listdir(src)
    train_images_count.append(len(current_dir))
    for image in current_dir:
        new_src = os.path.join(src,image)
        img_arr = np.array(img.imread(new_src))
        train_images.append(img_arr)
        

dir_list = sorted(os.listdir(path[1]))
dir_list.pop(0)
val_images = []
val_images_count = []
for curr_path in dir_list:
    src = os.path.join(path[1],curr_path)
    current_dir = os.listdir(src)
    val_images_count.append(len(current_dir))
    for image in current_dir:
        new_src = os.path.join(src,image)
        img_arr = np.array(img.imread(new_src))   
        val_images.append(img_arr)
        
train_x = []
for image in train_images:
    edge_img = canny(image)
    mid_pixel = len(image)//2
    train_img = []
    new_img = image[mid_pixel-25:mid_pixel+25]
    for i in new_img[::10]:
        train_img.append(i[(len(i)//2)-30:(len(i)//2)+30])
    mid_pixel = len(edge_img)//2
    new_img = edge_img[mid_pixel-25:mid_pixel+25]
    for i in new_img[::10]:
        train_img.append(i[(len(i)//2)-30:(len(i)//2)+30])
    train_img = np.array(train_img)
    train_img = np.reshape(train_img,10*60)
    train_x.append(train_img)
train_x = np.array(train_x)
train_x = train_x.reshape(len(train_x),-1)

test_x = []
for image in val_images:
    edge_img = canny(image)
    mid_pixel = len(image)//2
    test_img = []
    new_img = image[mid_pixel-25:mid_pixel+25]
    for i in new_img[::10]:
        test_img.append(i[(len(i)//2)-30:(len(i)//2)+30])
    mid_pixel = len(edge_img)//2
    new_img = edge_img[mid_pixel-25:mid_pixel+25]
    for i in new_img[::10]:
        test_img.append(i[(len(i)//2)-30:(len(i)//2)+30])
    test_img = np.array(test_img)
    test_img = np.reshape(test_img,10*60)
    test_x.append(test_img)
test_x = np.array(test_x)
test_x = test_x.reshape(len(test_x),-1)

train_y = []
value = 0
labels = [1,2,3,4,9,5,6,7,8,10]
for i in train_images_count:
    for j in range(i):
        train_y.append(labels[value])
    value+=1
train_y = np.array(train_y)

test_y = []
value = 0
for i in val_images_count:
    for j in range(i):
        test_y.append(labels[value])
    value+=1

clf = MLPClassifier()
clf.fit(train_x,train_y)

result = clf.predict(train_x)
print(result)
accuracy = 0
for i in range(len(result)):
    accuracy+=1 if result[i]==train_y[i] else 0
print(accuracy/len(result))

result = clf.predict(test_x)
print(result)
accuracy = 0
for i in range(len(result)):
    accuracy+=1 if result[i]==test_y[i] else 0
print(accuracy/len(result))
    
# accuracy = 0
# for i in range(len(result)):
#     for j in range(len(result[i])): 
#         if result[i][j]!=train_y[i][j]:
#             accuracy-=1
#             break
#     accuracy+=1 
# print(accuracy/len(result))

# result = clf.predict(test_x)
# print(result)
# accuracy = 0
# for i in range(len(result)):
#     for j in range(len(result[i])): 
#         if result[i][j]!=test_y[i][j]:
#             accuracy-=1
#             break
#     accuracy+=1 
# print(accuracy/len(result))
    
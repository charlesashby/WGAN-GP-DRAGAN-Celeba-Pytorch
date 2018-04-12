import csv


def split_line(line):
    data = []
    line = line.split(' ')
    for l in line:
        if l != '':
            data.append(l)
    return data[0], data[1:]

with open('data/list_attr_celeba.txt', 'r') as f:
    lines = f.readlines()




f = open('data/list_attr_celeba.txt', 'r')
lines = f.readlines()
n_lines = lines[0]
features = lines[1].split(' ')
feature = 'Eyeglasses'
# hair_feature = ['Bold', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
# hair_feature_id = [features.index(f) for f in hair_feature]
# hair_feature_id = [8, 9, 11, 17]
feature_id = features.index(feature)
X = []
images = []



with open('data_annotation_eyeglasses.csv', 'a') as f:
    writer = csv.writer(f)
    for line in lines[2:]:
        line = split_line(line)
        images.append(line[0])
        X.append(int(line[1][feature_id] == '1'))
        writer.writerow([line[0], int(line[1][feature_id] == '1')])


# cp data
import glob
import shutil
import os

PATH = '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/img_align_celeba'
images = glob.glob(PATH + "/*.jpg")

valid_imgs = images[170000:]
test_imgs = images[165000:]

for img in valid_imgs:
    shutil.copy(img, '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/img_valid/%s' % img.split('/')[-1])



for img in test_imgs:
    shutil.copy(img, '/home/ashbylepoc/PycharmProjects/gan-tutorial/data/img_test/%s' % img.split('/')[-1])

for img in images[150000:]:
    os.remove(img)
import os
import numpy as np
import json
import pprint

# image_path = './data/images'
annotation_path = './data/captions_train2014.json'
pp = pprint.PrettyPrinter()


f = open(annotation_path, encoding='utf-8')
data = json.load(f)
images = data['images']
annos = data['annotations']
pp.pprint(images[0])
pp.pprint(annos[0])
print(len(images))
print(len(annos))

f.close()

# images = images.sort(key='file_name')
images = sorted(images, key=lambda image : image['file_name'])
print("after sorted")
pp.pprint(images[0])
# exit(233)

imageid_to_anno = {}
for anno in annos:
    image_id = anno['image_id']
    if image_id not in imageid_to_anno.keys():
        imageid_to_anno[image_id] = [anno]
    else:
        imageid_to_anno[image_id].append(anno)


with open('results_COCO2014.token', 'w') as fp:
    cnt = 0
    for img in images:
        img_annos = imageid_to_anno[img['id']]
        # img_annos = []
        # for anno in annos:
        #     if anno['image_id'] == img['id']:
        #         img_annos.append(anno)
        anno_num = len(img_annos)
        if anno_num < 5:
            print("Error!!! img " + img['file_name'] + ' has ' + str(anno_num) + ' annotations!')
            continue
        for i in range(5):
            anno = img_annos[i]
            line = img['file_name'] + '#' + str(i) + '\t' + anno['caption']
            line = line.strip('\n')
            if line[-1] != '.' and line[-2] != '.' and line[-3] != '.':
                line += ' .'
            elif line[-1] == '.' and line[-2] != ' ':
                line = line[:-1] + ' .'
            elif line[-2] == '.' and line[-3] != ' ':
                line = line[:-2] + ' .'
            if len(line) > 30:
                fp.write(line + '\n')
            cnt += 1
        if cnt % 50000 == 0:
            print("processing " + str(cnt) + " annos")

    print('Total cnt of annos (adopted):', cnt)


f1 = open('results_COCO2014.token', 'r')
f2 = open('results_COCO2014_(2).token', 'w')
for line in f1:
    if '.jpg' not in line:
        continue
    f2.write(line)
f1.close()
f2.close()


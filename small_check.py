import numpy as np


feats = np.load('./data/feats_resnet50_COCO2014.npy')

print(feats[1])
print(feats[28])

cnt = 0
for i in range(10000):
    if (feats[i] == feats[1]).all():
    # if identical(feats[i], feats[1]):
        cnt += 1
print(cnt, "/10000")


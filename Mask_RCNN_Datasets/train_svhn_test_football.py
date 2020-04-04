from os import listdir
import json

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset

from numpy import zeros
from numpy import asarray

from matplotlib import pyplot

class SVHNDataset(Dataset):
    def load_dataset(self):
        for i in range(10):
            self.add_class("svhn", i if i != 0 else 10, str(i))
        images_dir = 'svhn/train/'
        for filename in listdir(images_dir):
            if filename[-4:] != ".png":
                continue
            image_id = filename[:-4]
            img_path = images_dir + filename
            self.add_image('svhn', image_id=int(image_id), path=img_path)
        with open('svhn/svhn_train_annot.json') as f:
            annotData = json.load(f)
        self.annotObj = {}
        for a in annotData["data"]:
            self.annotObj[a["name"]] = a

    def extract_boxes(self, filename):
        fAnnot = self.annotObj[filename]
        boxes = list()
        class_ids = list()
        for box in fAnnot["bbox"]:
            xmin = box["left"]
            ymin = box["top"]
            xmax = xmin + box["width"]
            ymax = ymin + box["height"]
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            class_ids.append(box["label"] if box["label"] != 0 else 10)
        return boxes, class_ids, fAnnot["width"], fAnnot["height"]

    def load_mask(self, image_id):
        fname = str(self.image_info[image_id]["id"]) + ".png"
        boxes, class_ids, w, h = self.extract_boxes(fname)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
        return masks, asarray(class_ids, dtype='int32')
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

class FootballDataset(Dataset):
    def load_dataset(self, is_train=True):
        for i in range(10):
            self.add_class('football', i if i != 0 else 10, str(i))
        images_dir = 'data/person_proposals/'

        with open('data/football_train_test_split.json') as f:
            football_train_test_split = json.load(f)
        images = football_train_test_split["train" if is_train else "test"]
        
        for filename in images:
            image_id = filename[:-4]
            img_path = images_dir + filename
            self.add_image('football', image_id=image_id, path=img_path)

        with open('data/jersey_number_labelling_via_project.json') as f:
            viaProject = json.load(f)
        self.annotObj = {}
        for key in viaProject["_via_img_metadata"]:
            self.annotObj[viaProject["_via_img_metadata"][key]["filename"]] = viaProject["_via_img_metadata"][key]

    def extract_boxes(self, filename):
        fAnnot = self.annotObj[filename]
        boxes = list()
        class_ids = list()
        for region in fAnnot["regions"]:
            xmin = region["shape_attributes"]["x"]
            ymin = region["shape_attributes"]["y"]
            xmax = xmin + region["shape_attributes"]["width"]
            ymax = ymin + region["shape_attributes"]["height"]
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            region_label = int(region["region_attributes"]["digit"])
            class_ids.append(region_label if region_label != 0 else 10)
        return boxes, class_ids, 256, 256 # all the football images are 256x256

    def load_mask(self, image_id):
        fname = self.image_info[image_id]["id"] + ".jpg"
        boxes, class_ids, w, h = self.extract_boxes(fname)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
        return masks, asarray(class_ids, dtype='int32')
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# train set
train_set = FootballDataset()
train_set.load_dataset()
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# load an image
image_id = 45
train_set.image_reference(image_id)
image = train_set.load_image(image_id)
print(image.shape)
# # load image mask
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)
# # plot image
pyplot.imshow(image)
# # plot mask
pyplot.imshow(mask[:, :, 1], cmap='gray', alpha=0.5)
pyplot.show()
import os
import sys

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import download_trained_weights

from FootballDataset import FootballDataset
from SVHNDataset import SVHNDataset

if (len(sys.argv) < 2):
    sys.exit("1 cmd line argument required: train set [svhn|football]")

if sys.argv[1] != 'svhn' and sys.argv[1] != 'football':
    sys.exit("Invalid train arg. Got " + sys.argv[1] + ", expected svhn or football")

# train set and validation set
if sys.argv[1] == 'svhn':
    train_set = SVHNDataset()
    train_set.load_dataset(is_train=True)
    validation_set = SVHNDataset()
    validation_set.load_dataset(is_train=False)
else:
    train_set = FootballDataset()
    train_set.load_dataset(subset='train')
    validation_set = FootballDataset()
    validation_set.load_dataset(subset='val')
train_set.prepare()
validation_set.prepare()

class FootballConfig(Config):
    NAME = 'football_cfg'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    NUM_CLASSES = 1 + 10 # background + digits
    IMAGE_MIN_DIM = 256
    STEPS_PER_EPOCH = 6681 if sys.argv[1] == 'svhn' else 384
    VALIDATION_STEPS = 1670 if sys.argv[1] == 'svhn' else 96
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

config = FootballConfig()
config.display()

model = MaskRCNN(mode='training', model_dir='./', config=config)

# Local path to trained weights file
if (len(sys.argv) == 3):
    model.load_weights(sys.argv[2], by_name=True)
else:
    COCO_MODEL_PATH = "./mask_rcnn_coco.h5"
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        download_trained_weights(COCO_MODEL_PATH)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(train_set, validation_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


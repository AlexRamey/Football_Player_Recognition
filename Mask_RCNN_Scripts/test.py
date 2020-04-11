import sys

from FootballDataset import FootballDataset

from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

if (len(sys.argv) != 2):
    sys.exit("1 cmd line arg expected specifying weights file to use")

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

test_set = FootballDataset()
test_set.load_dataset(subset='test')
test_set.prepare()

class FootballPredictionConfig(Config):
    NAME = 'football_prediction_cfg'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 10 # background + digits
    IMAGE_MIN_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

config = FootballPredictionConfig()
config.display()

model = MaskRCNN(mode='inference', model_dir='./', config=config)

model.load_weights(sys.argv[1], by_name=True)

test_mAP = evaluate_model(test_set, model, config)
print("Test mAP: %.3f" % test_mAP)

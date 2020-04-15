# Author: Alex Ramey
# Date: April 12, 2020
# detector.py holds a detector class that detects football player
# jersey numbers in images. It works in a 2-stage process. First,
# it runs a COCO pretrained mask rcnn on the input to generate
# person proposals, then it passes each of these through a separately
# trained mask rcnn that specializes in jersey number detection.
# It returns the set of bounding boxes, class names, and scores.

import json
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import download_trained_weights
import numpy as np
import os
from PIL import Image

class CocoConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class DigitConfig(Config):
    NAME = 'digit_cfg'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 10 # background + digits
    IMAGE_MIN_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

class Detector:
    def __init__(self, stage1_confidence_threshold=0.95, stage2_confidence_threshold=0.85, debug=False, verbose=False):
        self.stage1_confidence_threshold = stage1_confidence_threshold
        self.stage2_confidence_threshold = stage2_confidence_threshold
        self.debug = debug
        self.debugCounter = 0
        self.verbose = verbose

        self.stage1 = MaskRCNN(mode='inference', model_dir='.', config=CocoConfig())
        COCO_MODEL_PATH = "mask_rcnn_coco.h5"
        if not os.path.exists(COCO_MODEL_PATH):
            download_trained_weights(COCO_MODEL_PATH)
        self.stage1.load_weights(COCO_MODEL_PATH, by_name=True)
        self.stage1_class_names = ['BG', 'person']

        self.stage2 = MaskRCNN(mode='inference', model_dir='.', config=DigitConfig())
        WEIGHTS_PATH = os.path.join('training', 's5f15', 'svhn_0004_football_0014.h5')
        self.stage2.load_weights(WEIGHTS_PATH, by_name=True)
        self.stage2_class_names = ['BG', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    
    def detect(self, frame):
        stage1_results = self.stage1.detect([frame], verbose=self.verbose)

        if self.debug:
            self.debugCounter += 1
            Image.fromarray(frame).save(os.path.join('debug', 'frame_' + str(self.debugCounter) + '.jpg'))

        people, transforms = self.getPeopleFromStage1Results(frame, stage1_results[0])

        if self.debug:
            count = 1
            for person in people:
                Image.fromarray(person).save(os.path.join('debug', 'frame_' + str(self.debugCounter) + '_person_' + str(count) + '.jpg'))
                count += 1

        stage2_results = {'rois': [], 'class_names': [], 'scores': []}
        for i, person in enumerate(people):
            result = self.stage2.detect([person], verbose=self.verbose)[0]
            if any(score >= self.stage2_confidence_threshold for score in result['scores']):
                result['class_names'] = list(map(lambda id: self.stage2_class_names[id], result['class_ids']))
                result = self.transformStage2ResultsToOriginalCoordinateFrame(result, transforms[i])
                stage2_results['rois'] += result['rois']
                stage2_results['class_names'] += result['class_names']
                stage2_results['scores'] += result['scores']

        if self.debug:
            stage2_results_copy = {'rois': [], 'class_names': [], 'scores': []}
            with open(os.path.join('debug', 'frame_' + str(self.debugCounter) + '_digits.json'), 'w') as outfile:
                for roi in stage2_results['rois']:
                    stage2_results_copy['rois'].append(list(map(lambda x: float(x), roi)))
                stage2_results_copy['class_names'] = stage2_results['class_names']
                stage2_results_copy['scores'] = list(map(lambda x: float(x), stage2_results['scores']))
                json.dump(stage2_results_copy, outfile)

        return stage2_results

    def getPeopleFromStage1Results(self, frame, stage1_results):
        people, transforms = [], []
        for i, id in enumerate(stage1_results["class_ids"]):
            if id != self.stage1_class_names.index('person'):
                continue
            if stage1_results["scores"][i] < self.stage1_confidence_threshold:
                continue

            y1, x1, y2, x2 = stage1_results["rois"][i]
            y1 = int(max(0, y1))
            x1 = int(max(0, x1))
            y2 = int(min(frame.shape[0], y2))
            x2 = int(min(frame.shape[1], x2))

            # extract person bounding box region
            extracted_region = frame[y1:y2, x1:x2, :].copy()

            height = y2 - y1
            width = x2 - x1
            top_row_pad, btm_row_pad, left_col_pad, right_col_pad = 0, 0, 0, 0
            if height > width:
                left_col_pad = int((height - width) / 2)
                right_col_pad = height - width - left_col_pad
            elif width > height:
                top_row_pad = int((width - height) / 2)
                btm_row_pad = width - height - left_col_pad

            # pad to square shape
            extracted_region = np.pad(extracted_region, ((top_row_pad, btm_row_pad), (left_col_pad, right_col_pad), (0, 0)), 'constant')
            
            # save this information so that the transforms can be undone
            transforms.append([y1, x1, top_row_pad, left_col_pad, extracted_region.shape])
            
            # resize to 256x256
            extracted_region = np.array(Image.fromarray(extracted_region).resize((256,256)))
            people.append(extracted_region)

        return people, transforms

    def transformStage2ResultsToOriginalCoordinateFrame(self, result, transform):
        original_shape = transform[4]
        scale_factor = original_shape[0] / 256.0
        
        # undo scaling
        scaled_rois = list(map(lambda roi: list(map(lambda e: int(e * scale_factor), roi)), result['rois']))
        
        # undo padding
        for i in range(len(scaled_rois)):
            scaled_rois[i][0] -= transform[2] 
            scaled_rois[i][2] -= transform[2] 
            scaled_rois[i][1] -= transform[3] 
            scaled_rois[i][3] -= transform[3]

        # undo translation
        for i in range(len(scaled_rois)):
            scaled_rois[i][0] += transform[0] 
            scaled_rois[i][2] += transform[0] 
            scaled_rois[i][1] += transform[1] 
            scaled_rois[i][3] += transform[1]

        retVal = {'rois': [], 'class_names': [], 'scores': []}

        for i in range(len(scaled_rois)):
            if result['scores'][i] >= self.stage2_confidence_threshold:
                retVal['rois'].append(scaled_rois[i])
                retVal['class_names'].append(result['class_names'][i])
                retVal['scores'].append(result['scores'][i])

        return retVal
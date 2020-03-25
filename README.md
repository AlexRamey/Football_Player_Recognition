# Football Player Recognition

This project will identify the jersey numbers of American football players in broadcast footage using a two stage approach. The first stage will be a pre-trained Mask R-CNN that will detect players, and the second stage will be a fine-tuned Faster R-CNN to extract digits from the player bounding boxes.

## Dataset 
1. Capture frames from two Alabama football games (1080 x 720 resolution), one with home uniforms and the other with away uniforms.

1. Do a pass over the frames, removing ones that do not contain at least one Alabama player with a completely visible jersey number.

1. Run a [pre-trained Mask R-CNN](https://github.com/matterport/Mask_RCNN) to extract person bounding boxes from the frames.

1. Pad the bounding boxes with 0s (blackness) to create square images then re-scale to size 256 x 256. Discard non-Alabama players.

1. Label the digits with [VGG Image Annotator (VIA) tool](http://www.robots.ox.ac.uk/~vgg/software/via/).

1. Divide the dataset using a 70/30 split into `football_player_train` and `football_player_test`.

## Training

1. Start with the same [pre-trained Mask R-CNN](https://github.com/matterport/Mask_RCNN) as above. As shown in this [tutorial](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/), it is possible to train the Mask R-CNN for object detection using a dataset that only contains bounding boxes.

1. Evaluate performance on `football_player_test` for the following three scenarios:
    
    1. Fine-tuning with [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/) dataset
    1. Fine-tuning with `football_player_train`
    1. Fine-tuning with `SVHN + football_player_train` 

3. The best performing model is hereby referred to as `jersey_number_detector`.

## Inference

Perform inference on broadcast footage as follows:

```python
while (frame = video.getFrame()):
    player_boxes = preTrainedMaskRCNN.process(frame)
    player_boxes = list(map(lambda b: pad_and_scale(b), player_boxes)) 
    digits = jersey_number_detector.process(player_boxes)
    output(digits)
```

## Post-processing

1. Filter out away team jersey numbers based on color, which will vary per game
1. Supplement the jersey numbers with roster information (player names)
1. Filter out sideline player noise

## Next Steps

1. Add game context using OCR on the scoreboard
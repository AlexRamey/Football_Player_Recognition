import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import matplotlib.pyplot as plt

import csv
import json
import time

def display_image(image):
  fig = plt.figure(figsize=(1280, 720))
  plt.grid(False)
  plt.imshow(image)

def draw_bounding_box_on_image(image,
                               left,
                               right,
                               top,
                               bottom,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  # im_width, im_height = image.size
  # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
  #                               ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

# boxes should be a list of (left, right, top, bottom) tuples
# class names should be a list of strings
# scores should be a list of percentages
# image should be a numpy representation of a PIL Image
def draw_boxes(image, boxes, class_names, scores):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())
  font = ImageFont.load_default()

  for i in range(len(boxes)):
    top = boxes[i]["top"]
    bottom = boxes[i]["bottom"]
    left = boxes[i]["left"]
    right = boxes[i]["right"]

    display_str = "{}: {}%".format(class_names[i], int(scores[i]))
    color = colors[hash(class_names[i]) % len(colors)]
    #image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box_on_image(
        image,
        left,
        right,
        top,
        bottom,
        color,
        font,
        display_str_list=[display_str])
  return image

def map_box(b):
    b["right"] =  b["left"] + b["width"]
    b["bottom"] = b["top"] + b["height"]
    return b

def map_box_to_class(b):
    return b["label"]

with open('util/mTurk_Digit_Results.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_name = (row['Input.image_url']).rsplit('/')[-1]
        boxes = json.loads(row['Answer.annotatedResult.boundingBoxes'])

        im = Image.open("data/person_proposals/" + img_name)

        classes = list(map(map_box_to_class, boxes))
        boxes = list(map(map_box, boxes))
        scores = [1 for _ in range(0, len(boxes))]

        image_with_boxes = draw_boxes(im.copy(), boxes, classes, scores)
        print("Showing Img")
        image_with_boxes.show()
        time.sleep(5)



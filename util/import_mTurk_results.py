import csv
import json

with open('util/player_numbers.json') as f:
    viaProject = json.load(f)

filenamesToKeys = {}
for key in viaProject["_via_img_metadata"]:
    filenamesToKeys[viaProject["_via_img_metadata"][key]["filename"]] = key

def map_box_to_region(b):
    return {
        "shape_attributes": {
            "name": "rect",
            "x": b["left"],
            "y": b["top"],
            "width": b["width"],
            "height": b["height"]
        },
        "region_attributes": {
            "digit": b["label"]
        }
    }

with open('util/mTurk_Digit_Results.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_name = (row['Input.image_url']).rsplit('/')[-1]
        boxes = json.loads(row['Answer.annotatedResult.boundingBoxes'])
        regions = list(map(map_box_to_region, boxes))
        viaProject["_via_img_metadata"][filenamesToKeys[img_name]]["regions"] = regions

with open('util/player_numbers_out.json', 'w') as outfile:
    json.dump(viaProject, outfile)

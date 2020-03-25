import csv
import json

with open('data/jersey_number_labelling_via_project.json') as f:
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

def getRegionArea(region):
    return int(region["shape_attributes"]["width"]) * int(region["shape_attributes"]["height"])

# checkWork() decides whether or not to pay a worker based on how close
# their labels were to the final labels achieved after manual post-processing.
# Many workers did a good job, but there were also a substantial number of workers
# that would put a box around all digits of a jersey number and label it
# a nonsensical value like '0' or '4' when in reality it was an '89' that should
# have been labelled with 2 bounding boxes, one '8' and another '9'.
# Note: this method errors on the side of generosity . . .
# Returns a feedback string that is non-empty for rejections and empty for approvals.
def checkWork(workerSubmission, finalResultAfterManualPostProcessing):
    # Check 1: Reject work if not a single digit label is retained
    isSingleLabelRetained = False
    for regionProposal in workerSubmission:
        proposedDigit = regionProposal["region_attributes"]["digit"]
        for region in finalResultAfterManualPostProcessing:
            if region["region_attributes"]["digit"] == proposedDigit:
                isSingleLabelRetained = True
    if not isSingleLabelRetained:
        return 'No digits were accurately labelled'

    # Check 2: Reject work where the worker submitted a single bounding box
    # but the final result included exactly 2 bounding boxes whose area was
    # roughly equivalent to the area of the original single box
    if len(workerSubmission) == 1 and len(finalResultAfterManualPostProcessing) == 2:
        workerSubmissionArea = getRegionArea(workerSubmission[0])
        totalFinalArea = getRegionArea(finalResultAfterManualPostProcessing[0]) + getRegionArea(finalResultAfterManualPostProcessing[1])
        ratio = workerSubmissionArea * 1.0 / totalFinalArea
        if (abs(1-ratio) <= 0.3):
            return 'Entire jersey number was boxed instead of individual digits'

    return '' # Approve


with open('data/mTurk/workerFeedback.csv', 'w', newline='') as feedbackFile:
    with open('data/mTurk/mTurk_Digit_Results.csv', newline='') as submissionFile:
        reader = csv.DictReader(submissionFile)
        writer = csv.DictWriter(feedbackFile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            img_name = (row['Input.image_url']).rsplit('/')[-1]
            boxes = json.loads(row['Answer.annotatedResult.boundingBoxes'])
            workerSubmission = list(map(map_box_to_region, boxes))

            if img_name not in filenamesToKeys:
                row['Approve'] = 'x'
                writer.writerow(row)
                continue
            
            finalResultAfterManualPostProcessing = viaProject["_via_img_metadata"][filenamesToKeys[img_name]]["regions"]

            feedback = checkWork(workerSubmission, finalResultAfterManualPostProcessing)
            if len(feedback) == 0:
                row['Approve'] = 'x'
            else:
                row['Reject'] = feedback
            writer.writerow(row)

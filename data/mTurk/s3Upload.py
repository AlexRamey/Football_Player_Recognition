import boto3
import json
import os

s3 = boto3.resource('s3')
bucket = s3.Bucket('cs691-football-dataset')

with open('data/jersey_number_labelling_via_project.json') as f:
    data = json.load(f)
file_names = list(map(lambda val: val["filename"], data["_via_img_metadata"].values()))

for fname in file_names:
    bucket.upload_file(os.path.join("data/person_proposals", fname), fname, ExtraArgs={'ACL': 'public-read'})
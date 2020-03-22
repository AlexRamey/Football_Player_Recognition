import boto3

s3 = boto3.resource('s3')
bucket = s3.Bucket('cs691-football-dataset')

count = 0
for obj in bucket.objects.all():
    count += 1

print("Count: " + str(count))
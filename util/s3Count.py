import boto3
bucket_name = 'cs691-football-dataset'
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

count = 0
for obj in bucket.objects.all():
    # Uncomment to get all the urls
    # print(f'https://{bucket_name}.s3.amazonaws.com/{obj.key}')
    
    count += 1

print("Count: " + str(count))
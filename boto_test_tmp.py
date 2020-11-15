import boto3

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id='AKIASDWXX2IZTRXQUT6P',
    aws_secret_access_key='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'
)

for bucket in s3.buckets.all():
    print(bucket.name)
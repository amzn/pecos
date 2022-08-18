from aws_cdk import aws_iam as iam
import json 

def get_policies(accountID):
    #       Create necessary policies
    f=open('pecos_batch_job_multi_node/user_input.json')
    data = json.load(f)
    user_input_identifier = str(data["identifier"])
    f.close()
    #ECR policy
    policy_statement_1 = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "EcrRead",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                "Resource": "arn:aws:ecr:*:"+accountID+":repository/core-pecos-build-release-multi-node-test"
            },
            {
                "Sid": "EcrAuthorize",
                "Effect": "Allow",
                "Action": "ecr:GetAuthorizationToken",
                "Resource": "*"
            }
        ]
    }
    #output policy 
    policy_statement_2 = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "OutputBucket",
                "Action": [
                    "s3:Get*",
                    "s3:Put*",
                    "s3:List*"
                ],
                "Effect": "Allow",
                "Resource": [
                    "arn:aws:s3:::"+accountID+"-"+user_input_identifier+"-core-pecos-multi-node-bucket",
                    "arn:aws:s3:::"+accountID+"-"+user_input_identifier+"-core-pecos-multi-node-bucket/*"
                ]
            }
        ]
    }

    parse_json = iam.PolicyDocument.from_json
    return [parse_json(policy_statement_1), parse_json(policy_statement_2)]


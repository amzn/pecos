from aws_cdk import aws_iam
from aws_cdk import Stack


class PecosDistributedIAMStack(Stack):
    def __init__(self, scope, construct_id, param_config, storage_stack, ecr_stack, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.ecs_instance_role, self.ecs_instance_profile = self.create_ecs_instance_role(param_config.user_name)
        self.batch_job_role = self.create_batch_job_role(
            param_config.user_name,
            storage_stack.s3_bucket.bucket_name,
            ecr_stack.ecr_assets.repository.repository_arn
        )

    def create_ecs_instance_role(self, user_name):
        ecs_instance_role = aws_iam.Role(
            self,
            id="PECOS-Distributed-Ecs-Instance-Role",
            role_name=f"PECOS-Distributed-Ecs-Instance-Role-{user_name}",
            assumed_by=aws_iam.CompositePrincipal(aws_iam.ServicePrincipal("ec2.amazonaws.com")),
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                ),
            ]
        )
        ecs_instance_profile = aws_iam.CfnInstanceProfile(
            self,
            id="PECOS-Distributed-Ecs-Instance-Profile",
            instance_profile_name=f"PECOS-Distributed-Ecs-Instance-Profile-{user_name}",
            roles=[ecs_instance_role.role_name]
        )
        return ecs_instance_role, ecs_instance_profile

    def create_batch_job_role(self, user_name, s3_bucket_name, erc_repo_arn):
        job_role = aws_iam.Role(
            self,
            id="PECOS-Distributed-Batch-Job-Role",
            assumed_by=aws_iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            role_name=f"PECOS-Distributed-Batch-Role-{user_name}",
            description="Job role for Batch PECOS Inference.",
            inline_policies={
                "ECR-Read-Policy": self.create_ecr_read_policy(erc_repo_arn),
                "User-S3-Read-Write-Policy": self.create_s3_rw_policy(s3_bucket_name)
            },
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess")
            ]
        )
        return job_role

    @classmethod
    def create_ecr_read_policy(cls, erc_repo_arn):
        policy_json = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "EcrRead",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage"
                    ],
                    "Resource": erc_repo_arn
                },
                {
                    "Sid": "EcrAuthorize",
                    "Effect": "Allow",
                    "Action": "ecr:GetAuthorizationToken",
                    "Resource": "*"
                }
            ]
        }
        return aws_iam.PolicyDocument.from_json(policy_json)

    @classmethod
    def create_s3_rw_policy(cls, s3_bucket_name):
        s3_bucket_arn = f"arn:aws:s3:::{s3_bucket_name}"
        policy_json = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "S3BucketReadWrite",
                    "Action": [
                        "s3:Get*",
                        "s3:Put*",
                        "s3:List*"
                    ],
                    "Effect": "Allow",
                    "Resource": [
                        s3_bucket_arn,
                        s3_bucket_arn + "/*"
                    ]
                }
            ]
        }
        return aws_iam.PolicyDocument.from_json(policy_json)

from aws_cdk import aws_s3
from aws_cdk import aws_efs
from aws_cdk import aws_ec2
from aws_cdk import aws_iam
from aws_cdk import Stack
from aws_cdk import RemovalPolicy


class PecosDistributedStorageStack(Stack):
    def __init__(self, scope, construct_id, param_config, vpc_stack, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.s3_bucket = self.create_s3_bucket(
            param_config.account,
            param_config.user_name
        )
        self.shared_disk = self.create_efs(
            vpc_stack.vpc,
            vpc_stack.security_group,
            param_config.user_name
        )

    def create_s3_bucket(self, account, user_name):
        return aws_s3.Bucket(
            self,
            id="PECOS-Distributed-Bucket",
            bucket_name=f"pecos-distributed-bucket-{account}-{user_name}",
            block_public_access=aws_s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY
        )

    def create_efs(self, vpc, security_group, user_name):
        shared_efs = aws_efs.FileSystem(
            self,
            id="PECOS-Distributed-EFS",
            file_system_name=f"PECOS-Distributed-EFS-{user_name}",
            vpc=vpc,
            vpc_subnets=aws_ec2.SubnetSelection(
                subnet_type=aws_ec2.SubnetType.PRIVATE_WITH_NAT
            ),
            security_group=security_group,
            enable_automatic_backups=False,
            encrypted=False,
            performance_mode=aws_efs.PerformanceMode.GENERAL_PURPOSE,
            removal_policy=RemovalPolicy.DESTROY
        )
        shared_efs.node.default_child.file_system_policy = aws_iam.PolicyDocument(
            statements=[
                aws_iam.PolicyStatement(
                    effect=aws_iam.Effect.ALLOW,
                    principals=[aws_iam.AnyPrincipal()],
                    actions=[
                        "elasticfilesystem:ClientMount",
                        "elasticfilesystem:ClientWrite",
                        "elasticfilesystem:ClientRootAccess",
                    ],
                    conditions={"Bool": {"elasticfilesystem:AccessedViaMountTarget": "true"}},
                )
            ]
        )
        return shared_efs

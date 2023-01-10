import os
from aws_cdk import Stack
from aws_cdk import aws_ecr_assets


class PecosDistributedEcrStack(Stack):
    def __init__(self, scope, construct_id, param_config, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.ecr_assets = aws_ecr_assets.DockerImageAsset(
            self,
            id="PECOS-Distributed-Ecr-Image",
            directory=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../../../"
            ),
            file=os.path.join(
                "aws_infra/multinode_batch_cdk/cdk_constructs",
                "dockerfile",
                "Dockerfile"
            )
        )

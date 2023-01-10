#!/usr/bin/env python3
import aws_cdk
import sys
import os
from cdk_constructs.batch import PecosDistributedBatchStack
from cdk_constructs.vpc import PecosDistributedVPCStack
from cdk_constructs.iam import PecosDistributedIAMStack
from cdk_constructs.storage import PecosDistributedStorageStack
from cdk_constructs.ecr import PecosDistributedEcrStack
from cdk_constructs.param_config import PecosDistributedParamConfig


try:
    param_config = PecosDistributedParamConfig.from_json(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "param_config.json"))
except FileNotFoundError:
    raise FileNotFoundError(
        f"Configuration json: 'param_config.json' not found. "
        f"Please run './config_generator.py' to generate."
    )

cdk_env = aws_cdk.Environment(account=param_config.account, region=param_config.region)

app = aws_cdk.App()

vpc_stack = PecosDistributedVPCStack(
    app,
    "PecosDistributedVPCStack",
    param_config,
    stack_name=f"PecosDistributedVPCStack-{param_config.user_name}",
    env=cdk_env
)

storage_stack = PecosDistributedStorageStack(
    app,
    "PecosDistributedStorageStack",
    param_config,
    vpc_stack,
    stack_name=f"PecosDistributedStorageStack-{param_config.user_name}",
    env=cdk_env
)

ecr_stack = PecosDistributedEcrStack(
    app,
    "PecosDistributedEcrStack",
    param_config,
    stack_name=f"PecosDistributedEcrStack-{param_config.user_name}",
    env=cdk_env
)

iam_stack = PecosDistributedIAMStack(
    app,
    "PecosDistributedIAMStack",
    param_config,
    storage_stack,
    ecr_stack,
    stack_name=f"PecosDistributedIAMStack-{param_config.user_name}",
    env=cdk_env
)

batch_stack = PecosDistributedBatchStack(
    app,
    "PecosDistributedBatchStack",
    param_config,
    vpc_stack,
    storage_stack,
    iam_stack,
    ecr_stack,
    stack_name=f"PecosDistributedBatchStack-{param_config.user_name}",
    env=cdk_env
)

aws_cdk.Tags.of(app).add("CREATEDBY", param_config.user_name)
app.synth()

#!/usr/bin/env python3
import os

import aws_cdk as cdk

from pecos_batch_job_multi_node.pecos_batch_job_multi_node_stack import PecosBatchJobMultiNodeStack

accountID = str(os.environ.get('CDK_DEFAULT_ACCOUNT'))
accountRegion = str(os.environ.get("CDK_DEFAULT_REGION"))
app = cdk.App()
PecosBatchJobMultiNodeStack(app, "PecosBatchJobMultiNodeStack",
    env=cdk.Environment(account=accountID, region=accountRegion)

    )

app.synth()

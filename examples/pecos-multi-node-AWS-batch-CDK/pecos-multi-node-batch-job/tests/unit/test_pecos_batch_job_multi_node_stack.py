import aws_cdk as core
import aws_cdk.assertions as assertions

from pecos_batch_job_multi_node.pecos_batch_job_multi_node_stack import PecosBatchJobMultiNodeStack

# example tests. To run these tests, uncomment this file along with the example
# resource in pecos_batch_job_multi_node/pecos_batch_job_multi_node_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = PecosBatchJobMultiNodeStack(app, "PECOS-Batch-job-multi-node")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })

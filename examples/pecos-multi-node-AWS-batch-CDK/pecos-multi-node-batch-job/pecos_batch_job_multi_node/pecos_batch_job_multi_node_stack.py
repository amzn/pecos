from http import client
from platform import platform
from time import time
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    RemovalPolicy,
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_batch_alpha as batch,
    aws_batch as cfnbatch
)

import os
import boto3
import json
from .get_policies import get_policies


class PecosBatchJobMultiNodeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
#       Get user's input
        f = open('./pecos_batch_job_multi_node/user_input.json')
        data = json.load(f)
        user_input_disk = int(data["disk"])
        user_input_memory = int(data["memory"])
        user_input_vCPU = int(data["vCPU"])
        user_input_identifier = str(data["identifier"])
        user_input_key = str(data["key"])
        user_input_value = str(data["value"])
        user_input_num_nodes = int(data['nodes'])
        f.close()


#       Get user's accout ID
        accountID = str(os.environ.get('CDK_DEFAULT_ACCOUNT'))
        # TODO: Change bucket name if necessary 
        custom_bucket_name = str(accountID) + \
             "-"+user_input_identifier+"-core-pecos-a2q-test-bucket"
#       check if need to create a S3 bucket, if yes, create one for users
        if self.check_bucket_existence(custom_bucket_name) == False:
            bucket = s3.Bucket(self, "Pecos-Bucket",
                               bucket_name=custom_bucket_name,
                               block_public_access=s3.BlockPublicAccess.BLOCK_ALL)


#       Create an role
        batch_role = iam.Role(self, "PECOS-Batch-Role-Multi-Node",
                              assumed_by=iam.ServicePrincipal(
                                  "ecs-tasks.amazonaws.com"),
                              role_name="PECOS-Batch-Role-Multi-Node-"+user_input_identifier,
                              description="This is a custom role for PECOS-Batch-Multi-Node.",
                              )
        
        batch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"))

#       prepare policy documents and attach to the role
        my_custom_policy_1, my_custom_policy_2 = get_policies(accountID)
        new_managed_policy_1 = iam.CfnManagedPolicy(self, "PECOS-ECR-Image",
                                                    policy_document=my_custom_policy_1,
                                                    roles=["PECOS-Batch-Role-Multi-Node-"+user_input_identifier],
                                                    )
        new_managed_policy_1.node.add_dependency(batch_role)
        new_managed_policy_2 = iam.CfnManagedPolicy(self, "Input-Output-S3-bucket",
                                                    policy_document=my_custom_policy_2,
                                                    roles=["PECOS-Batch-Role-Multi-Node-"+user_input_identifier],
                                                    )
                                
        new_managed_policy_2.node.add_dependency(batch_role)
        batch_role.apply_removal_policy(RemovalPolicy.DESTROY)
#       Create VPC for this CDK
        vpc = self.custom_vpc()
        #Create security group 
        security_group = ec2.SecurityGroup(
            self, "batch-job-security-group", vpc=vpc, allow_all_outbound=True)
        security_group.node.add_dependency(vpc)
        #Allow hosts inside the security group to connect to each other
        security_group.connections.allow_internally(ec2.Port.all_traffic(
        ), "Allow hosts inside the security group to connect to each other.")
        #Create template
        template = self.launch_template(
            security_group, user_input_disk, user_input_identifier)
        template.node.add_dependency(vpc)
        #Create compute environment
        compute_environment = self.compute_environment(
            accountID, vpc, security_group, user_input_identifier)

        compute_environment.node.add_dependency(template)
        #Create Job queue
        job_queue = self.job_queue(compute_environment, user_input_identifier)

        job_queue.node.add_dependency(compute_environment)
        #Create job definition
        job_definition = self.job_definition(
            accountID, user_input_memory, user_input_vCPU, user_input_identifier, user_input_num_nodes)
#       Assign all constructs a tag to identify 
        cdk.Tags.of(self).add(user_input_key, user_input_value)


#   Check if Pecos-bucket exist
    def check_bucket_existence(self, BucketName):
        s3 = boto3.resource('s3')
        if s3.Bucket(BucketName) in s3.buckets.all():
            return True
        else:
            return False


    def custom_vpc(self):
        vpc = ec2.Vpc(self,
                      id="PECOS-multi-node-VPC",
                      cidr="10.3.0.0/16",
                      #Every region has at least 3 azs
                      max_azs=3,
                      nat_gateways=1,
                      subnet_configuration=[
                          ec2.SubnetConfiguration(
                              name="public", cidr_mask=24,
                              reserved=False, subnet_type=ec2.SubnetType.PUBLIC),
                          ec2.SubnetConfiguration(
                              name="private", cidr_mask=24,
                              reserved=False, subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT),
                      ],
                      enable_dns_hostnames=True,
                      enable_dns_support=True)
        return vpc


    def launch_template(self, security_group, user_input_disk, user_input_identifier):
        multipart_user_data = ec2.MultipartUserData()
        commands_user_data = ec2.UserData.for_linux()
        #Add user data for template 
        multipart_user_data.add_user_data_part(
            commands_user_data, ec2.MultipartBody.SHELL_SCRIPT, True)
        commands_user_data.add_commands("sudo mkfs -t xfs /dev/xvdf")
        commands_user_data.add_commands("sudo mkdir /data")
        commands_user_data.add_commands("sudo mount /dev/xvdf /data")

        template = ec2.LaunchTemplate(self, "PECOS-Batch-launch-template-multi-node-CDKtest",
                                      launch_template_name="PECOS-Batch-launch-template-multi-node-CDKtest-" +
                                      user_input_identifier,
                                      block_devices=[ec2.BlockDevice(
                                          device_name="/dev/xvdf",
                                          volume=ec2.BlockDeviceVolume.ebs(user_input_disk, delete_on_termination=True,
                                                                           encrypted=False, volume_type=ec2.EbsDeviceVolumeType.GP2),)],
                                      security_group=security_group,
                                      user_data=multipart_user_data,
                                      )

        return template

    def compute_environment(self, accountID, vpc, security_group, user_input_identifier):
        compute_environment = batch.ComputeEnvironment(self, "PECOS-Batch-compute-environment-multi-node-CDKtest",
                                                       compute_environment_name="PECOS-Batch-compute-environment-multi-node-CDKtest-" +
                                                       user_input_identifier,
                                                       compute_resources=batch.ComputeResources(
                                                           vpc=vpc,
                                                           type=batch.ComputeResourceType.ON_DEMAND,
                                                           maxv_cpus=256,
                                                           vpc_subnets=ec2.SubnetSelection(
                                                               subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT),
                                                           allocation_strategy=batch.AllocationStrategy.BEST_FIT,
                                                           launch_template=batch.LaunchTemplateSpecification(
                                                               launch_template_name="PECOS-Batch-launch-template-multi-node-CDKtest-" +
                                                               user_input_identifier,
                                                               version="$Latest",
                                                           ),

                                                           minv_cpus=0,
                                                           security_groups=[
                                                               security_group],

                                                       ),
                                                       service_role=iam.Role.from_role_arn(self, "batch-service-role",
                                                                                           "arn:aws:iam::"+accountID+":role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch"),
                                                       )

        return compute_environment

    def job_queue(self, compute_environment, user_input_identifier):
        job_queue = batch.JobQueue(self, "PECOS-Batch-job-queue-multi-node-CDKtest",
                                   compute_environments=[batch.JobQueueComputeEnvironment(
                                       compute_environment=compute_environment,
                                       order=1)],
                                   priority=1,
                                   job_queue_name="PECOS-Batch-job-queue-multi-node-CDKtest-"+user_input_identifier,
                                   )
        return job_queue

    def job_definition(self, accountID, user_input_memory, user_input_vCPU, user_input_identifier, user_input_num_nodes):
        cfn_job_definition = cfnbatch.CfnJobDefinition(self, "PECOS-Batch-job-definition-multi-node-CDKtest",
                                                       type="multinode",
                                                       job_definition_name="PECOS-Batch-job-definition-multi-node-CDKtest-" +
                                                       user_input_identifier,
                                                       node_properties=cfnbatch.CfnJobDefinition.NodePropertiesProperty(
                                                           main_node=0,
                                                           num_nodes=user_input_num_nodes,
                                                           node_range_properties=[cfnbatch.CfnJobDefinition.NodeRangePropertyProperty(
                                                               target_nodes="0:",
                                                               container=cfnbatch.CfnJobDefinition.ContainerPropertiesProperty(
                                                                #Image ID is fixed. User has to store image under ECR with name core-pecos-build-release-multi-node-test 
                                                                    image=accountID+".dkr.ecr.us-east-1.amazonaws.com/core-pecos-build-release-multi-node-test",
                                                                   execution_role_arn="arn:aws:iam::"+accountID + \
                                                                   ":role/PECOS-Batch-Role-Multi-Node-"+user_input_identifier,
                                                                   job_role_arn="arn:aws:iam::"+accountID+":role/PECOS-Batch-Role-Multi-Node-"+user_input_identifier,
                                                                   memory=user_input_memory*1000,
                                                                   mount_points=[cfnbatch.CfnJobDefinition.MountPointsProperty(
                                                                       container_path="/data",
                                                                       read_only=False,
                                                                       source_volume="data")],
                                                                   privileged=True,
                                                                   user="ecs-user",
                                                                   vcpus=user_input_vCPU,
                                                                   volumes=[cfnbatch.CfnJobDefinition.VolumesProperty(
                                                                       host=cfnbatch.CfnJobDefinition.VolumesHostProperty(
                                                                           source_path="/data"
                                                                       ),
                                                                       name="data"
                                                                   )]
                                                               )

                                                           )]
                                                       )
                                                       )
        return cfn_job_definition

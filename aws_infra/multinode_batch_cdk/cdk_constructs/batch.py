from aws_cdk import Stack
from aws_cdk import aws_ec2
from aws_cdk import aws_batch


class PecosDistributedBatchStack(Stack):
    _SHARED_DISK_MOUNT_NAME = "shared_data"

    def __init__(
        self,
        scope,
        construct_id,
        param_config,
        vpc_stack,
        storage_stack,
        iam_stack,
        ecr_stack,
        **kwargs
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.launch_template = self.create_launch_template(
            vpc_stack.security_group,
            storage_stack.shared_disk,
            param_config.user_disk_gb_req,
            param_config.user_name
        )

        self.compute_environment = self.create_compute_environment(
            vpc_stack.vpc,
            vpc_stack.security_group,
            iam_stack.ecs_instance_profile,
            self.launch_template,
            param_config.user_name
        )

        self.job_queue = self.create_job_queue(self.compute_environment, param_config.user_name)

        self.job_definition = self.create_job_definition(
            ecr_stack.ecr_assets.image_uri,
            iam_stack.batch_job_role,
            param_config.user_num_node,
            param_config.user_mem_gb_req,
            param_config.user_cpu_req,
            param_config.user_name)

    def create_launch_template(self, security_group, shared_disk, user_disk_gb_req, user_name):
        update_cmd = aws_ec2.UserData.for_linux()
        update_cmd.add_commands(
            "sudo yum update -y",
            "sudo yum install -y amazon-efs-utils",
            f"sudo mkdir -p /{self._SHARED_DISK_MOUNT_NAME}",
            f"sudo mount -t efs -o tls {shared_disk.file_system_id}:/ /{self._SHARED_DISK_MOUNT_NAME}",
            f"sudo chmod 777 /{self._SHARED_DISK_MOUNT_NAME}"
        )

        multipart_user_data = aws_ec2.MultipartUserData()
        multipart_user_data.add_user_data_part(
            user_data=update_cmd,
            content_type=aws_ec2.MultipartBody.SHELL_SCRIPT
        )

        return aws_ec2.LaunchTemplate(
            self,
            id="PECOS-Distributed-Launch-Template",
            launch_template_name=f"PECOS-Distributed-Launch-Template-{user_name}",
            block_devices=[
                aws_ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=aws_ec2.BlockDeviceVolume.ebs(
                        volume_size=user_disk_gb_req,
                        delete_on_termination=True,
                        encrypted=False,
                        volume_type=aws_ec2.EbsDeviceVolumeType.GP2
                    )
                )
            ],
            user_data=multipart_user_data
        )

    def create_compute_environment(self, vpc, security_group, instance_profile, launch_template, user_name):
        return aws_batch.CfnComputeEnvironment(
            self,
            id="PECOS-Distributed-Compute-Environment",
            compute_environment_name=f"PECOS-Distributed-Compute-Environment-{user_name}",
            type="MANAGED",
            state="ENABLED",
            compute_resources=aws_batch.CfnComputeEnvironment.ComputeResourcesProperty(
                type="EC2",
                maxv_cpus=999,
                minv_cpus=0,
                desiredv_cpus=0,
                instance_role=instance_profile.attr_arn,
                instance_types=["optimal", "c5", "m5", "r5", "x1"],
                allocation_strategy="BEST_FIT",
                subnets=vpc.select_subnets(
                    subnet_type=aws_ec2.SubnetType.PRIVATE_WITH_NAT
                ).subnet_ids,
                security_group_ids=[security_group.security_group_id],
                launch_template=aws_batch.CfnComputeEnvironment.LaunchTemplateSpecificationProperty(
                    launch_template_id=launch_template.launch_template_id,
                    version="$Latest"
                ),
                update_to_latest_image_version=True
            )
        )

    def create_job_queue(self, compute_environment, user_name):
        return aws_batch.CfnJobQueue(
            self,
            id="PECOS-Distributed-Batch-Job-Queue",
            job_queue_name=f"PECOS-Distributed-Batch-Job-Queue-{user_name}",
            state="ENABLED",
            compute_environment_order=[
                aws_batch.CfnJobQueue.ComputeEnvironmentOrderProperty(
                    compute_environment=compute_environment.attr_compute_environment_arn,
                    order=1
                )
            ],
            priority=1
        )

    def create_job_definition(self, ecr_image_uri, batch_job_role, user_num_node, user_mem_gb_req, user_cpu_req, user_name):
        return aws_batch.CfnJobDefinition(
            self,
            id="PECOS-Distributed-Batch-Job-Definition",
            job_definition_name=f"PECOS-Distributed-Batch-Job-Definition-{user_name}",
            type="multinode",
            node_properties=aws_batch.CfnJobDefinition.NodePropertiesProperty(
                main_node=0,
                num_nodes=user_num_node,
                node_range_properties=[
                    aws_batch.CfnJobDefinition.NodeRangePropertyProperty(
                        target_nodes="0:",
                        container=aws_batch.CfnJobDefinition.ContainerPropertiesProperty(
                            image=ecr_image_uri,
                            job_role_arn=batch_job_role.role_arn,
                            user="ecs-user",
                            privileged=True,
                            mount_points=[
                                aws_batch.CfnJobDefinition.MountPointsProperty(
                                    container_path=f"/{self._SHARED_DISK_MOUNT_NAME}",
                                    read_only=False,
                                    source_volume=f"{self._SHARED_DISK_MOUNT_NAME}"
                                )
                            ],
                            volumes=[
                                aws_batch.CfnJobDefinition.VolumesProperty(
                                    host=aws_batch.CfnJobDefinition.VolumesHostProperty(
                                        source_path=f"/{self._SHARED_DISK_MOUNT_NAME}"
                                    ),
                                    name=f"{self._SHARED_DISK_MOUNT_NAME}"
                                )
                            ],
                            resource_requirements=[
                                aws_batch.CfnJobDefinition.ResourceRequirementProperty(
                                    type="VCPU", value=str(user_cpu_req)
                                ),
                                aws_batch.CfnJobDefinition.ResourceRequirementProperty(
                                    type="MEMORY", value=str(user_mem_gb_req * 1024)
                                )
                            ],
                            ulimits=[
                                aws_batch.CfnJobDefinition.UlimitProperty(
                                    hard_limit=-1,
                                    name="memlock",
                                    soft_limit=-1
                                )
                            ]
                        )
                    )
                ]
            )
        )

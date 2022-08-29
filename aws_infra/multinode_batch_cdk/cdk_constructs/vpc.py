from aws_cdk import Stack
from aws_cdk import aws_ec2


class PecosDistributedVPCStack(Stack):
    def __init__(self, scope, construct_id, param_config, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.vpc = self.create_vpc(param_config.user_name)
        self.security_group = self.create_security_group(self.vpc, param_config.user_name)

    def create_vpc(self, user_name):
        vpc = aws_ec2.Vpc(
            self,
            id="PECOS-Distributed-VPC",
            vpc_name=f"PECOS-Distributed-VPC-{user_name}",
            cidr="10.0.0.0/16",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                aws_ec2.SubnetConfiguration(
                    name="public", cidr_mask=24,
                    reserved=False, subnet_type=aws_ec2.SubnetType.PUBLIC
                ),
                aws_ec2.SubnetConfiguration(
                    name="private", cidr_mask=24,
                    reserved=False, subnet_type=aws_ec2.SubnetType.PRIVATE_WITH_NAT
                ),
            ],
            enable_dns_hostnames=True,
            enable_dns_support=True
        )
        return vpc

    def create_security_group(self, vpc, user_name):
        security_group = aws_ec2.SecurityGroup(
            self,
            id="PECOS-Distributed-Security-Group",
            security_group_name=f"PECOS-Distributed-SG-{user_name}",
            vpc=vpc,
            allow_all_outbound=True
        )
        security_group.connections.allow_internally(
            port_range=aws_ec2.Port.all_traffic(),
            description="Allow hosts inside the security group to connect to each other."
        )
        return security_group

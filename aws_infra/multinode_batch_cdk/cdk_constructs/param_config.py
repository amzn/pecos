import json


class PecosDistributedParamConfig(object):
    """
    Parameters for PECOS distributed jobs
    """

    def __init__(
        self,
        account,
        region,
        user_name,
        user_disk_gb_req
    ):
        self.account = account
        self.region = region
        self.user_name = user_name
        self.user_disk_gb_req = user_disk_gb_req
        # Default value for generating multi-node batch constructs
        # Overridable at submitting job
        self.user_num_node = 2
        self.user_mem_gb_req = 350
        self.user_cpu_req = 1

    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            param_dict = json.load(f)
            return cls(**param_dict)

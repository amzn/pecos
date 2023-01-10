#!/usr/bin/env python3

import argparse
import json
import logging
import os
import boto3


class PecosBatchJobArgs(object):
    """Class to parse and store PECOS multi-node Batch job arguments."""

    def __init__(self):
        self._configure_attr(self._get_args())

    @classmethod
    def _configure_attr(self, args):
        """Configure self attributes from parsed args"""
        # Copied From args
        args_dict = vars(args)
        for key, val in args_dict.items():
            setattr(self, key, val)

        # Created from args
        with open(args.cdk_config) as f:
            cdk_conf_dict = json.load(f)
        self.account = cdk_conf_dict["account"]
        self.region = cdk_conf_dict["region"]
        self.user_name = cdk_conf_dict["user_name"]
        self.job_definition = f"PECOS-Distributed-Batch-Job-Definition-{self.user_name}"
        self.job_queue = f"PECOS-Distributed-Batch-Job-Queue-{self.user_name}"
        self.input_s3_arn = f"s3://pecos-distributed-bucket-{self.account}-{self.user_name}/{args.input_folder}/"
        self.output_s3_arn = f"s3://pecos-distributed-bucket-{self.account}-{self.user_name}/{args.output_folder}/"

    @classmethod
    def _get_args(self):
        """Get Batch job args"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cdk-config",
            metavar="CDK_CONF_PATH",
            type=str,
            default=f"{os.path.join(os.path.dirname(os.path.realpath(__file__)), 'param_config.json')}",
            help="CDK parameter configuration file path"
        )
        parser.add_argument(
            "--job-name",
            metavar="JOB_NAME",
            type=str,
            required=True,
            help="AWS Batch Job Name"
        )
        parser.add_argument(
            "--input-folder",
            metavar="INPUT",
            type=str,
            required=True,
            help="S3 input folder name"
        )
        parser.add_argument(
            "--output-folder",
            metavar="OUTPUT",
            type=str,
            required=True,
            help="S3 output folder name"
        )
        parser.add_argument(
            "--num-nodes",
            metavar="NUM_NODES",
            type=int,
            required=True,
            help="Number of nodes for multi-node processing Batch jobs"
        )
        parser.add_argument(
            "--cpu",
            metavar="CPU_COUNTS",
            type=int,
            required=True,
            help=f"Number of vCPU for each node"
        )
        parser.add_argument(
            "--memory",
            metavar="MEM_SIZE",
            type=int,
            required=True,
            help=f"Memory size in megabytes for each node"
        )
        parser.add_argument(
            "--commands",
            metavar="COMMANDS",
            required=True,
            type=str,
            help=f"Commands to be executed on main node"
        )
        parser.add_argument(
            "--batch-bootstrap-timeout-min",
            metavar="BATCH_BOOTSTRAP_TIMEOUT_MIN",
            type=int,
            default=30,
            help="Timeout(min) for Batch nodes to finish joining main node and bootstrapping"
        )
        return parser.parse_args()


class PecosBatchJobSubmitter(object):
    """Class to submit PECOS Multi-node Processing Batch Jobs.
    """

    _WORKSPACE = "/pecos_workspace"
    _JOB_SCRIPT = "run_mnp_job.sh"

    def __init__(self, pecos_batch_job_args):
        if not isinstance(pecos_batch_job_args, PecosBatchJobArgs):
            raise ValueError(type(pecos_batch_job_args))
        self._args = pecos_batch_job_args
        self._batch_client = boto3.client("batch", region_name=self._args.region)

    def _prepare_workspace_cmd(self):
        """Prepare workspace folder command"""
        return [
            f"sudo mkdir -p {self._WORKSPACE}",
            f"sudo chown -R $USER:amazon {self._WORKSPACE}",
            f"cd {self._WORKSPACE}",
            f"mkdir -p {self._args.input_folder}",
            f"mkdir -p {self._args.output_folder}",
            f"export PECOS_WORKSPACE={self._WORKSPACE}",
            f"export PECOS_INPUT={os.path.join(self._WORKSPACE, self._args.input_folder)}",
            f"export PECOS_OUTPUT={os.path.join(self._WORKSPACE, self._args.output_folder)}"
        ]

    def _download_input_s3_cmd(self):
        """Download input from s3 command"""
        return [
            f"aws s3 cp --quiet --recursive {self._args.input_s3_arn} $PECOS_INPUT"
        ]

    def _upload_output_s3_cmd(self):
        """Upload output to S3 command"""
        return [
            f"aws s3 cp --quiet --recursive $PECOS_OUTPUT {self._args.output_s3_arn}"
        ]

    def _cleanup_cmd(self):
        """Clean up command"""
        return ["sudo rm -rf $PECOS_BUILDS_WORKSPACE"]

    def _main_node_job_cmd(self, job_cmd):
        """Job execution command only run on main node.

        For multi-node procesing jobs, need to write commands into a bash script on disk,
        and set as env BATCH_ENTRY_SCRIPT for scheduler to execute.
        The job command consists of doing work and uploading output,
        and only main node needs to execute this bash script.
        """
        job_script_path = os.path.join(self._WORKSPACE, self._JOB_SCRIPT)
        job_cmd = "\n".join(["#!/bin/bash", "set -e"] + job_cmd)
        commands = [
            f"echo '{job_cmd}' > {job_script_path}",
            f"cat {job_script_path}",
            f"chmod 755 {job_script_path}",
            f"export BATCH_ENTRY_SCRIPT={job_script_path}",
            f"export BATCH_BOOTSTRAP_TIMEOUT={self._args.batch_bootstrap_timeout_min}",
            f"/batch-runtime-scripts/entry-point.sh",
        ]
        return commands

    def _get_batch_job_spec(self, commands):
        """Create a Multi-node processing Batch job spec"""
        return {
            "jobName": self._args.job_name,
            "jobQueue": self._args.job_queue,
            "jobDefinition": self._args.job_definition,
            "nodeOverrides": {
                "numNodes": self._args.num_nodes,
                "nodePropertyOverrides": [
                    {
                        "targetNodes": "0:",
                        "containerOverrides": {
                            "command": ["bash", "-c", " && ".join(commands)],
                            "resourceRequirements": [
                                {"type": "VCPU", "value": str(self._args.cpu)},
                                {"type": "MEMORY", "value": str(self._args.memory)},
                            ],
                        },
                    }
                ],
            },
        }

    def submit(self):
        """Submit Batch job"""
        # Assemble batch job commands
        commands = []
        commands += self._prepare_workspace_cmd()
        commands += self._download_input_s3_cmd()
        commands += self._main_node_job_cmd(
            ["pwd", "cd $PECOS_BUILDS_WORKSPACE && ls", self._args.commands] + self._upload_output_s3_cmd()
        )
        commands += self._cleanup_cmd()

        # Batch job specs
        batch_job_spec = self._get_batch_job_spec(commands)

        batch_job_spec_json = json.dumps(batch_job_spec, indent=2, sort_keys=True)
        logging.info(batch_job_spec_json)

        job_id = self._batch_client.submit_job(**batch_job_spec)["jobId"]
        logging.info(f"Submitted {self._args.job_name} to job queue {self._args.job_queue} - {job_id}")


if __name__ == "__main__":
    """
    Prerequisite
    ------------
    1. Setup AWS credentials.
    2. Upload input data to the folder created in PECOS distributed S3 bucket


    Sample Commands
    ---------------
    ./submit_job.py \
    --job-name pecos-train-xlinear-eurlex-4k \
    --input-folder input-eurlex-4k \
    --output-folder output-eurlex-4k \
    --num-nodes 2 \
    --cpu 1 \
    --memory 60000 \
    --commands 'mpiexec -n $AWS_BATCH_JOB_NUM_NODES -f /job/hostfile python3 -m pecos.distributed.xmc.xlinear.train \
        -x $PECOS_INPUT/X.trn.npz \
        -y $PECOS_INPUT/Y.trn.npz \
        -m $PECOS_OUTPUT/eurlex_model \
        --nr-splits 2 -b 50 -k 100 -nst 16 -t 0.1

        python3 -m pecos.xmc.xlinear.predict \
        -x $PECOS_INPUT/X.tst.npz \
        -y $PECOS_INPUT/Y.tst.npz \
        -m $PECOS_OUTPUT/eurlex_model > $PECOS_OUTPUT/eurlex_score.txt'
    """
    logging.basicConfig(level=logging.INFO)

    pecos_batch_job_args = PecosBatchJobArgs()
    job_submitter = PecosBatchJobSubmitter(pecos_batch_job_args)
    job_submitter.submit()

import sys
import time
import boto3
from botocore.compat import total_seconds
import json 
import os 

region = str(input("Please input your region. For example: us-east-1\n"))


batch = boto3.client(
    service_name='batch',
    region_name=region,
    endpoint_url='https://batch.'+region+'.amazonaws.com')

cloudwatch = boto3.client(
    service_name='logs',
    region_name=region,
    endpoint_url='https://batch.'+region+'.amazonaws.com',
    )



def main():
    f=open('pecos_batch_job_multi_node/user_input.json')
    data = json.load(f)
    user_input_identifier = str(data["identifier"])
    user_input_num_nodes = int(data["nodes"])
    user_input_vCPU = int(data["vCPU"])
    user_input_memory = int(data["memory"])*1000
    f.close()

    f=open('pecos_batch_job_multi_node/job_command.json')
    data = json.load(f)
    user_input_prepare_data = str(data["prepare_data_input"])
    user_input_train_data_command = str(data["train_data_command"])
    user_input_predict_command = str(data["predict_command"])
    user_input_output_path = str(data["output_path"])
    f.close()

    logGroupName = '/aws/batch/job'

    jobName = "multi-node-CDKtest"
    jobQueue = "PECOS-Batch-job-queue-multi-node-CDKtest-"+user_input_identifier
    jobDefinition = "PECOS-Batch-job-definition-multi-node-CDKtest-"+user_input_identifier
    job_script_path = "/tmp/run_mnp_job.sh"
    

    job_cmd_1 = "\n".join(["#!/bin/bash", "set -e", "pwd", user_input_train_data_command, user_input_predict_command])
    job_cmd_2 = [
            f"cd /home/ecs-user && sudo chown ecs-user:ecs-user /home/ecs-user && sudo chmod ug+rw /home/ecs-user",
            f"{user_input_prepare_data}",
            f"echo '{job_cmd_1}' > {job_script_path}",
            f"cat {job_script_path}",
            f"chmod 755 {job_script_path}",
            f"export BATCH_ENTRY_SCRIPT={job_script_path}",
            f"/batch-runtime-scripts/entry-point.sh",
            f"aws s3 cp --recursive /home/ecs-user/ s3://{user_input_output_path}",
        ]
    commands = ["bash", "-c", " && ".join(job_cmd_2)]

    print("You job command is below:\n")
    print(commands)
    
    submitJobResponse = batch.submit_job(
        jobName=jobName,
        jobQueue=jobQueue,
        jobDefinition=jobDefinition,
        nodeOverrides={
        'numNodes': user_input_num_nodes,
        'nodePropertyOverrides': [
            {
                'targetNodes': '0:',
                'containerOverrides': {
                    'vcpus': user_input_vCPU,
                    'memory': user_input_memory,
                    'command': commands,
                }
            },
        ]
    },
    )

    jobId = submitJobResponse['jobId']
    print ('Submitted job [%s - %s] to the job queue [%s]' % (jobName, jobId, jobQueue))


    wait = True
    temp = ""
    while wait:
        for i in range(5):
            print(f'\r{"     " * 5 + "please wait"}', end = "", flush=True)
            print(f'\r{"......" * i}', end = "", flush=True)
            time.sleep(0.5)
        describeJobsResponse = batch.describe_jobs(jobs=[jobId])
        status = describeJobsResponse['jobs'][0]['status']
        
        if status == 'SUCCEEDED' or status == 'FAILED':
            print("\n")
            print ('%s' % ('=' * 80))
            print ('Job [%s - %s] %s' % (jobName, jobId, status))
            wait = False
            break
        elif status != temp:
            print ('\rJob [%s - %s] is %-9s' % (jobName, jobId, status)),
            sys.stdout.flush()
            temp = status
if __name__ == "__main__":
    main()
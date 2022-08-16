
# PECOS Multi-Node Batch job

This AWS CDK code deploys stack containing AWS batch constructs to your AWS account's CloudFormation. After you deploy this CDK on your account, you should have all the batch constructs needed to run a multi-node batch job in AWS for PECOS training. 

The development and deployment are performed on your AWS account's **EC2 instances with necessary permission**, please double-check your EC2 instance role policies if any permission issues occur.

## Prerequisites

Install npm:
```
# Download nvm script
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

#Copy and paste together
export NVM_DIR="$HOME/.nvm"                                                                              
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Install latest CDK-supported Node.js and npm
# NOTE: latest node version is not always supported by CDK, double-check before installation
nvm install 17.3.0

# Check installation
node -v
npm -v
```

Install AWS CDK:
```
npm install -g aws-cdk

# Check installation
cdk --version
```

Bootstrapping(Please make sure that your IAM user has CloudFormation full access):
```
# Replace contents in <> with your AWS account number and region
cdk bootstrap aws://<ACCOUNT_NUMBER>/<REGION>
```

## Build your own Docker image
Create a Amazon ECR with the name core-pecos-build-release-multi-node-test on console.  
Retrieve an authentication token and authenticate your Docker client to your registry.  
```
aws ecr get-login-password --region <Your region> | docker login --username AWS --password-stdin <Your account ID>.dkr.ecr.<Your region>.amazonaws.com
```

Build your Docker image using the following command.  
Docker file could be found within the DockerFile folder. You can skip this step if your image is already built:
```
docker build -t core-pecos-build-release-multi-node-test .
```

After the build completes, tag your image so you can push the image to this repository:
```
docker tag core-pecos-build-release-multi-node-test:latest <Your Accunt Id>.dkr.ecr.<Your region>.amazonaws.com/core-pecos-build-release-multi-node-test:latest
```

Run the following command to push this image to your newly created AWS repository:
```
docker push <Your Accunt Id>.dkr.ecr.<Your region>.amazonaws.com/core-pecos-build-release-multi-node-test:latest
```


## Usage
Clone code:
```
#Will avaliable soon 
#git clone https://github.com/amzn/pecos/examples


#Open the workspace 
cd pecos-multi-node-batch-job
```


Create Python virtual environment and install dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip install aws-cdk.aws-batch-alpha==2.31.0a0
python3 -m pip install boto3
```

Input your customize parameters:
```
#This step will ask for your choices of memory, disk, vCPUS, number of nodes. 
python3 ./pecos_batch_job_multi_node/get_parameter.py
```

CDK synthesize(Synthesize an AWS CloudFormation template for the app):
```
cdk synth
```

CDK diff (To see the changes):
```
cdk diff
```


CDK deploy all stacks(To deploy the stack using AWS CloudFormation):
```
cdk deploy --all
```

CDK destroy all stacks:
```
cdk destroy --all
```

Check if the deployed components works(optional):  

Please make sure that you already build and upload Docker image to your repository under the name of "core-pecos-build-release-multi-node-test".  
You can either go to AWS Batch console to submit job with those constructs you just generated via CDK directly or you can use provided job command assembler to genereate your command first: 
```
# This command will ask for commands to prepare, train, predict and save data. 
python3 ./pecos_batch_job_multi_node/job_command_assembler.py
```

Example:  
Distributed XLinear Model Training on eurlex-4k Data:
Check here for detail: https://github.com/amzn/pecos/tree/mainline/pecos/distributed/xmc/xlinear  
Command to prepare eurlex-4k data:
```
wget https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k.tar.gz && tar -zxvf eurlex-4k.tar.gz
```

Command to train:
```
mpiexec -n $AWS_BATCH_JOB_NUM_NODES -f /job/hostfile python3 -m pecos.distributed.xmc.xlinear.train -x ./xmc-base/eurlex-4k/tfidf-attnxml/X.trn.npz -y ./xmc-base/eurlex-4k/Y.trn.npz --nr-splits 2 -b 50 -k 100 -m eurlex_4k_model --min-n-sub-tree 16 -t 0.1 --meta-label-embedding-method pii --sub-label-embedding-method pifa --verbose-level 3
```
Command to predict:
```
python3 -m pecos.xmc.xlinear.predict -x ./xmc-base/eurlex-4k/tfidf-attnxml/X.tst.npz -y ./xmc-base/eurlex-4k/Y.tst.npz -m ./eurlex_4k_model
```

Command to save:
```
S3 output location to save data. 
<Your account number>-<Your identifier>-core-pecos-a2q-test-bucket
```



submit your job by running code below:
```
# This command will assemble your previous command input and trigger a job submission directly. 
python3 ./pecos_batch_job_multi_node/batch_job_testing.py
```

Job Command for reference for the above example: 
```
["bash","-c","cd /home/ecs-user && sudo chown ecs-user:ecs-user /home/ecs-user && sudo chmod ug+rw /home/ecs-user && wget https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k.tar.gz && tar -zxvf eurlex-4k.tar.gz && echo '#!/bin/bash\nset -e\npwd\nmpiexec -n $AWS_BATCH_JOB_NUM_NODES -f /job/hostfile python3 -m pecos.distributed.xmc.xlinear.train -x ./xmc-base/eurlex-4k/tfidf-attnxml/X.trn.npz -y ./xmc-base/eurlex-4k/Y.trn.npz --nr-splits 2 -b 50 -k 100 -m eurlex_4k_model --min-n-sub-tree 16 -t 0.1 --meta-label-embedding-method pii --sub-label-embedding-method pifa --verbose-level 3\npython3 -m pecos.xmc.xlinear.predict -x ./xmc-base/eurlex-4k/tfidf-attnxml/X.tst.npz -y ./xmc-base/eurlex-4k/Y.tst.npz -m ./eurlex_4k_model' > /tmp/run_mnp_job.sh && cat /tmp/run_mnp_job.sh && chmod 755 /tmp/run_mnp_job.sh && export BATCH_ENTRY_SCRIPT=/tmp/run_mnp_job.sh && /batch-runtime-scripts/entry-point.sh && aws s3 cp --recursive /home/ecs-user/ s3://<Your account number>-<Your identifier>-core-pecos-a2q-test-bucket"]
```







## References

* Developer Guide: https://docs.aws.amazon.com/cdk/v2/guide/home.html
* Python API Reference: https://docs.aws.amazon.com/cdk/api/v2/python/index.html


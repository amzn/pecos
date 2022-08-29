# PECOS Distributed Training Infra AWS Multi-node Batch CDK

This sub-folder contains AWS Multi-node Batch CDK code for fast generation of the infra for training distributed PECOS models.

## Prerequisite

1. AWS account with access to VPC, Batch, EC2, S3, ECR.
2. Setup AWS credentials following [[guide](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html)].
3. Install `npm` and `node`:
    ```
    # Download nvm script
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

    # Attach below to bash config (e.g. .zshrc)
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

    # Install latest CDK-supported Node.js and npm
    # NOTE: latest node version is not always supported by CDK, double-check before installation
    nvm install 14.6.0

    # Check installation
    node -v
    npm -v
    ```
4. Install AWS CDK:
    ```
    npm install -g aws-cdk

    # Check installation
    cdk --version
    ```

## Usage
Clone code:
```
git clone https://github.com/amzn/pecos.git
cd pecos/aws_infra/multinode_batch_cdk
```

Create Python virtual environment and install dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Configure parameters. Make sure *AWS account* and *AWS region* are the *same* as environment’s *AWS credential*.

```
./config_generator.py
```

Bootstrap:
```
# Replace contents in <> with your AWS account number and region
cdk bootstrap aws://<YOUR_AWS_ID>/<YOUR_REGION>
```

CDK synthesize and display change:
```
cdk synth
cdk diff
```

CDK deploy all stacks:
* An image for running distributed PECOS containers will also be built and uploaded, so it may take a while.
```
cdk deploy --all
```

If you do not need the infra anymore, destroy:
```
cdk destroy --all
```

## Example

Download `eurlex-4k` data and upload to the S3 bucket created by CDK:
```
wget https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k.tar.gz
tar -zxvf eurlex-4k.tar.gz
aws s3 cp --recursive ./xmc-base/eurlex-4k \
s3://pecos-distributed-bucket-<YOUR_AWS_ID>-<YOUR_NAME>/input-eurlex-4k/
```

Submit job by executing the following commands in the `aws_infra/multinode_batch_cdk` directory:
```
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
```

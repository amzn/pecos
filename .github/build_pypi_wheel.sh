#!/bin/bash
set -e

# Get pip
echo "Build wheel using Python version $PIP_VER..."
PIP=$(ls /opt/python/cp${PIP_VER//./}-cp*/bin/pip)
if [ -z $PIP ]; then
   echo "No pip found for version $PIP_VER, exit"
   exit 1
fi
echo "pip: $($PIP --version)"


# Install dependencies
echo "Install dependencies..."
$PIP install setuptools wheel twine auditwheel
yum install -y openblas-devel


# Build wheel
PECOS_SOURCE=$DOCKER_MNT/
WHEEL_OUTPUT_FOLDER=$DOCKER_MNT/$WHEEL_DIR

$PIP wheel $PECOS_SOURCE --no-deps -w $WHEEL_OUTPUT_FOLDER
WHEEL_NAME=$(ls $WHEEL_OUTPUT_FOLDER)

echo "Temporary wheel: $(ls $WHEEL_OUTPUT_FOLDER)"
auditwheel show $WHEEL_OUTPUT_FOLDER/$WHEEL_NAME

echo "Auditing wheel to platform $PLAT..."
auditwheel repair $WHEEL_OUTPUT_FOLDER/$WHEEL_NAME -w $WHEEL_OUTPUT_FOLDER
rm $WHEEL_OUTPUT_FOLDER/$WHEEL_NAME

echo "Audited wheel: $(ls $WHEEL_OUTPUT_FOLDER)"
auditwheel show $WHEEL_OUTPUT_FOLDER/$(ls $WHEEL_OUTPUT_FOLDER)
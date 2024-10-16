#!/bin/bash
set -e

# Git requires repository folder to be owned and accessed by the same user, or add the folder to safe directory
# Otherwise, git commands cannot be successfully executed and thus correct PECOS version could not be retrieved
# See: https://github.blog/2022-04-12-git-security-vulnerability-announced/
git config --global --add safe.directory /$DOCKER_MNT
PECOS_TAG=$(cd $DOCKER_MNT && git describe  --tags --abbrev=0)
echo "Building wheel for PECOS $PECOS_TAG..."

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

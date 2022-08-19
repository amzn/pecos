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
# Temporarily pin setuptools version due to a recent bug, which will be fix in their later updates
# See: https://github.com/pypa/setuptools/issues/3532
# https://github.com/numpy/numpy/issues/22135
$PIP install 'setuptools<=60.0.*' wheel twine auditwheel

# Install OpenBLAS
# Using Numpy pre-build OpenBLAS lib v0.3.19 hosted on Anaconda
# Refer to: https://github.com/MacPython/openblas-libs
# OpenBLAS64 is for ILP64, which is not our case
# Details see Numpy OpenBLAS downloader:
# https://github.com/numpy/numpy/blob/main/tools/openblas_support.py#L19
if [ "$PLAT" = "manylinux2014_x86_64" ] || [ "$PLAT" = "manylinux2014_aarch64" ]; then
   OPENBLAS_VER="v0.3.19-22-g5188aede"
   OPENBLAS_LIB="openblas-${OPENBLAS_VER}-${PLAT}.tar.gz"
   OPENBLAS_LIB_URL="https://anaconda.org/multibuild-wheels-staging/openblas-libs/$OPENBLAS_VER/download/$OPENBLAS_LIB"
   yum install wget -y
   wget $OPENBLAS_LIB_URL
   tar -xvf $OPENBLAS_LIB
else
   echo "$PLAT not supported."
   exit 1
fi


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

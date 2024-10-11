#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.


# Verbose tag
# Usage: pass VFLAG=-v to command line if need verbose mode.
# > make pecos VFLAG=-v
VFLAG ?=

# C/C++ Compiler warning as error flag, for code cleaning purposes
# Usage: pass WARN_AS_ERROR=True to command line if need to turn C/C++ warning into error
# > make libpecos WARN_AS_ERROR=True
# > make test WARN_AS_ERROR=True
WARN_AS_ERROR ?=
ifdef WARN_AS_ERROR
WARN_AS_ERROR_CMD = PECOS_MANUAL_COMPILE_ARGS="-Werror"
else
WARN_AS_ERROR_CMD =
endif

# Style and type checks
format: flake8 black mypy

flake8:
	@echo "Checking flake8 on PECOS source files..."
	python3 -m flake8 ${VFLAG} --config .github/style_type_check_cfg/.flake8 ./pecos
	@echo "Checking flake8 on PECOS test files..."
	python3 -m flake8 ${VFLAG} --config .github/style_type_check_cfg/.flake8 ./test

black:
	@echo "Checking black on PECOS source files..."
	python3 -m black ${VFLAG} --check --diff --config .github/style_type_check_cfg/.black ./pecos --exclude 'pecos/_version.py'
	@echo "Checking black on PECOS test files..."
	python3 -m black ${VFLAG} --check --diff --config .github/style_type_check_cfg/.black ./test

mypy:
	@echo "Checking mypy on PECOS source files..."
	python3 -m mypy ${VFLAG} --config-file .github/style_type_check_cfg/.mypy -p pecos
	@echo "Checking mypy on PECOS test files..."
	python3 -m mypy ${VFLAG} --config-file .github/style_type_check_cfg/.mypy `find ./test/ -type f -name "*.py"`


# Install and unit test
libpecos:
	python3 -m pip install --upgrade pip
	${WARN_AS_ERROR_CMD} python3 -m pip install ${VFLAG} --editable .

.PHONY: test
test: libpecos
	python3 -m pip install pytest pytest-coverage
	python3 -m pytest

# Clean
clean:
	rm ${VFLAG} -rf ./build ./dist ./*.egg-info
	rm -f ./pecos/core/*.so .coverage*
	rm -f ./pecos/_version.py
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

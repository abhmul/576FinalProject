#!/usr/bin/env bash

STRING="Running vector evaluator..."
PYTHON="/usr/bin/python3.5"
PROJECT_ROOT="/home/abhmul/PycharmProjects/576FinalProject/eval"
EVALUATE_SCRIPT="evaluate.py"

# Fixes import errors, see https://stackoverflow.com/questions/24727582/running-python-script-from-bash-file-causes-import-errors
pushd . > /dev/null 2>&1
cd ${PROJECT_ROOT}

echo ${STRING}

for vector_name in ../vectors/*.txt; do
    ${PYTHON} ${EVALUATE_SCRIPT} --filename ${vector_name} -n 99999999
done

popd > /dev/null 2>&1
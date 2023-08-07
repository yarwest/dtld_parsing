#!/bin/bash

set -e

# ============================================================
# CONFIGURATION VARIABLES
# ============================================================
SCRIPT_NAME=$(basename "$0")
MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

CONDA_ENV_NAME='tl-detection'
CONDA_BASE_PATH=$(conda info --base)

KAGGLE_JSON_PATH="${HOME}/.kaggle/kaggle.json"

DATASET_NAME="mbornoe/lisa-traffic-light-dataset"
DATASET_DIR="${MAIN_DIR}/data"

# ============================================================
# ARGUMENTS PARSING
# ============================================================
ARG_CONDA_ENV_NAME=''

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "${SCRIPT_NAME} - setup project with all dependencies"
      echo ""
      echo "This script creates a conda environment, installs all dependencies"
      echo "and downloads the dataset from Kaggle. The script uses mamba"
      echo "instead of conda if available."
      echo ""
      echo "${SCRIPT_NAME} [options]"
      echo ""
      echo "Options:"
      echo "-h, --help                show a brief help"
      echo "-e, --env=ENV_NAME        set conda environment name (default: ${CONDA_ENV_NAME})"
      exit 0
      ;;
    -e)
      shift
      if test $# -gt 0; then
        ARG_CONDA_ENV_NAME=$1
      else
        echo "Error: No environment name specified."
        exit 1
      fi
      shift
      ;;
    --env*)
      ARG_CONDA_ENV_NAME=$(echo "$1" | sed -e 's/^[^=]*=//g')
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n ${ARG_CONDA_ENV_NAME} ]]; then
  CONDA_ENV_NAME=${ARG_CONDA_ENV_NAME}
fi

# ============================================================
# DEPENDENCIES INSTALLATION
# ============================================================
echo "============================================================"
echo "1. Installing dependencies"
echo "============================================================"

if ! [[ -x "$(command -v conda)" || -x "$(command -v mamba)" ]]; then
  echo "Error: neither conda nor mamba is installed"
  echo "Use official Anaconda distribution or install Miniforge from"
  echo "https://github.com/conda-forge/miniforge"
  exit 1
fi

CONDA_CMD="conda"
source "${CONDA_BASE_PATH}/etc/profile.d/${CONDA_CMD}.sh"
if [[ -x "$(command -v mamba)" ]]; then
  CONDA_CMD="mamba"
  source "${CONDA_BASE_PATH}/etc/profile.d/${CONDA_CMD}.sh"
fi

# Create conda environment
echo "Creating ${CONDA_ENV_NAME} environment..."
set +e
yes | ${CONDA_CMD} env create -n ${CONDA_ENV_NAME} -f "${MAIN_DIR}/environment.yml" 2> /dev/null
CREATION_STATUS=$?
set -e

# If error occured it means that the env probably already exists
if [[ ${CREATION_STATUS} -eq 1 ]]; then
  echo "Environment ${CONDA_ENV_NAME} already exists. Updating..."
  yes | ${CONDA_CMD} env update -n ${CONDA_ENV_NAME} -f "${MAIN_DIR}/environment.yml"
fi

${CONDA_CMD} activate ${CONDA_ENV_NAME}

# ============================================================
# DATASET DOWNLOAD
# ============================================================
echo "============================================================"
echo "2. Downloading dataset"
echo "============================================================"

if [[ ! -f "${KAGGLE_JSON_PATH}" && (-z $KAGGLE_USERNAME || -z $KAGGLE_KEY )]]; then
    echo "Kaggle credentials not available. Please follow instructions at"
    echo "https://www.kaggle.com/docs/api#authentication to proceed."
    exit 1;
fi

kaggle datasets download -d "${DATASET_NAME}" -p "${DATASET_DIR}"
echo "Extracting archive..."
ARCHIVE_NAME=$(echo ${DATASET_NAME} | sed 's/.*\///')
unzip -qq -n "${DATASET_DIR}/${ARCHIVE_NAME}.zip" -d "${DATASET_DIR}/${ARCHIVE_NAME}"
echo "Extracted to ${DATASET_DIR}/${ARCHIVE_NAME}."

echo ""
echo "============================================================"
echo "Setup finished"
echo "============================================================"
echo "Activate your new environment using"
echo "$ ${CONDA_CMD} activate ${CONDA_ENV_NAME}"
echo "and install the package"
echo "$ pip install (-e) ."
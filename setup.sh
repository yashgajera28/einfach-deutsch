#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR="${VENV_DIR:-.venv}"

# Detect the operating system for activation and tesseract hints.
OS="unknown"
case "$(uname -s)" in
    Linux*)     OS="Linux" ;;
    Darwin*)    OS="Mac" ;;
    CYGWIN*|MINGW*|MSYS*) OS="Windows" ;;
esac

echo "Setting up Einfach Deutsch environment..."
echo "Detected OS: ${OS}"

# Ensure a Python interpreter is available.
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Error: ${PYTHON} not found. Please install Python 3.10+ and try again."
    exit 1
fi

PYTHON_VERSION=$(${PYTHON} --version 2>&1 | awk '{print $2}')
echo "Using Python ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist.
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment in ${VENV_DIR}..."
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "Using existing virtual environment in ${VENV_DIR}."
fi

# Activate virtual environment.
if [[ "${OS}" == "Windows" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/Scripts/activate"
else
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
fi

# Upgrade pip and install dependencies.
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download the German spaCy model (also listed in requirements.txt as a fallback).
echo "Downloading spaCy German model..."
python -m spacy download de_core_news_lg || true

# Create required runtime directories.
mkdir -p data/raw data/processed data/evaluation checkpoints outputs logs

echo ""
echo "Setup complete."

if [[ "${OS}" == "Windows" ]]; then
    echo "Activate with: ${VENV_DIR}\\Scripts\\activate"
else
    echo "Activate with: source ${VENV_DIR}/bin/activate"
fi

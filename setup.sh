#!/bin/sh
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"
ENV_ERROR="This module requires Python >=3.10, pip, and virtualenv to be installed."

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    # Viam's macOS cloud builder ships Python 3.9, too old for viam-sdk: get 3.11 through uv.
    echo "python3 is missing or older than 3.10; creating the virtualenv with uv and Python 3.11."
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || { echo $ENV_ERROR >&2; exit 1; }
    fi
    uv venv --seed --python 3.11 $VENV_NAME || { echo $ENV_ERROR >&2; exit 1; }
elif ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
    echo "Failed to create virtualenv."
    if command -v apt-get >/dev/null; then
        echo "Detected Debian/Ubuntu, attempting to install python3-venv automatically."
        SUDO="sudo"
        if ! command -v $SUDO >/dev/null; then
            SUDO=""
        fi
		if ! apt info python3-venv >/dev/null 2>&1; then
			echo "Package info not found, trying apt update"
			$SUDO apt -qq update >/dev/null
		fi
        $SUDO apt install -qqy python3-venv >/dev/null 2>&1
        if ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
            echo $ENV_ERROR >&2
            exit 1
        fi
    else
        echo $ENV_ERROR >&2
        exit 1
    fi
fi

# remove -U if viam-sdk should not be upgraded whenever possible
# -qq suppresses extraneous output from pip
echo "Virtualenv found/created. Installing/upgrading Python packages..."
if ! [ -f .installed ]; then
    if ! $PYTHON -m pip install -r requirements.txt -Uqq; then
        exit 1
    else
        touch .installed
    fi
fi

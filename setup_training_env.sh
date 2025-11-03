#!/bin/bash
set -e

# Install olmo-core from your commit with masking support
pip uninstall -y olmo-core
pip install git+https://github.com/allenai/OLMo-core.git@f18a9bf44496acc1fa0cad7d8c8b9fb111eff315

# Install other dependencies if needed
pip install -e ".[train]"

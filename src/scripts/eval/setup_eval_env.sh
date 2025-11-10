
pip install -e ".[eval]"
# git clone https://github.com/allenai/olmes.git
# cd olmes
# pip install -e .
# cd ..
# Override transformers and vllm versions, bypassing pyproject conflicts with olmes.
# TODO: remove once merged
# pip install vllm==0.7.0; pip uninstall -y transformers; pip install "transformers@git+https://github.com/swj0419/transformers"; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124; pip install ipdb;
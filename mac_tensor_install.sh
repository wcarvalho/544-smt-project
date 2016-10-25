conda install virtualenv
conda create --name=tf python=3.5
source activate tf
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl
pip install --ignore-installed --upgrade $TF_BINARY_URL
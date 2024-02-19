#!/bin/bash
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash /notebooks/Anaconda3-2023.09-0-Linux-x86_64.sh -b -p $HOME/anaconda3
activateBase () {
    . /root/anaconda3/bin/activate
}
activateBase
conda init
conda create -n newpy python=3.11
activateEnv (){
    conda activate newpy
}
activateEnv
pip install ipykernel jupyter ipywidgets jupyterlab
python -m ipykernel install --user --name newpy
python -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 torchtext==0.16.0+cpu torchdata==0.7.0 --index-url https://download.pytorch.org/whl/cu121torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install git+https://github.com/huggingface/transformers.git
python -m pip install pytorch-crf
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pipx install poetry
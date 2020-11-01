#!/usr/bin/env bash

set -x

intexit() {
    # Kill all subprocesses (all processes in the current process group)
    kill -HUP -$$
}

hupexit() {
    # HUP'd (probably by intexit)
    echo
    echo "Interrupted"
    exit
}

trap hupexit HUP
trap intexit INT

#wget "https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh"
#chmod +x Anaconda3-2018.12-Linux-x86_64.sh
#./Anaconda3-2018.12-Linux-x86_64.sh
#source ~/.bashrc

#conda install numpy -y
#conda install cudatoolkit==9.0  -y
#conda install tensorflow-gpu==1.11 -y
#pip install sumy
pip install pyrouge
#conda install spacy -y

conda uninstall spacy -y
pip uninstall spacy -y
conda uninstall spacy-nightly -y
pip uninstall spacy-nightly -y
conda uninstall thinc -y
pip uninstall thinc -y
conda uninstall regex -y
pip uninstall regex -y
yes | pip install regex==2018.01.10
yes | pip install thinc==6.12.0
yes | pip install spacy
yes | pip install spacy-stanfordnlp
yes | python -c 'import stanfordnlp
stanfordnlp.download("en")'
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
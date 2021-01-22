# se3-forcefield

Dependencies:

This repo uses the ANI-1 dataset/reader and the Deep Graph Library. To prepare dataset and libaries run the following from command line:

pip install --pre dgl-cu101
wget https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/90576
tar -xvf ANI1_release.tar.gz
export PYTHONPATH=${PYTHONPATH}:/${pwd}/ANI-1_release/readers/lib/

I've had problems with the PYTHONPATH line, so an alternative is (in python):

import sys
import os
cwd = os.getcwd()
sys.path.insert(1,os.path.join(cwd,'/ANI-1_release/readers/lib/'))


TODO:

It would be nice to have a setup.sh script to create/update a conda env that has all the dependencies.

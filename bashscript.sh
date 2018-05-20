sudo apt-get install python-virtualenv
sudo easy_install virtualenv
sudo pip install virtualenv
mkdir ~/virtualenvironment
virtualenv ~/virtualenvironment/my_new_app
cd ~/virtualenvironment/my_new_app/bin
source activate

pip install numpy
pip install pandas
pip install scikit-learn
pip install pickle
pip install seaborn
pip install h2o
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision

python Reduced_Model.py
python h2o.py
python Binary_2ndStage.py
python model_performance.py
python DeepOLS&SecondStage.py

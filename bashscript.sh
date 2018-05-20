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

python Reduced_Model.py
python h2o.py
python Binary_2ndStage.py
python DeepOLS&SecondStage.py

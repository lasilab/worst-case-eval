conda create -n SPODR python=3.9
source activate base
conda activate SPODR

pip install torch
pip install folktables
pip install matplotlib
pip install scikit-learn
pip install tqdm
conda install -c conda-forge pyomo
conda install -c conda-forge cyipopt
pip install numpy==1.23

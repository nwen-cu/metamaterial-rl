# GetFEM Installation on Clemson Palmetto Cluster

## Create Conda Environment

```bash
conda create -n metamtl-rl python=3.8
conda activate metamtl-rl
```

## Register Jupyter Kernel

```bash
conda install jupyter 
python -m ipykernel install --user --name metamtl-rl --display-name "Py3.8 MetaMaterial-RL"
```

## Install Python Dependency for GetFEM

```bash
conda install numpy scipy
```

## Create Libraries Directory

```bash
mkdir ~/scilib
cd ~/scilib
mkdir bin
export PATH=~/scilib/bin:$PATH
export C_INCLUDE_PATH=~/scilib/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/scilib/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/scilib/lib:$LIBRARY_PATH
```

## Create Build Space

```bash
mkdir ~/buildspace
cd ~/buildspace
```

## Install Qhull

```bash
cd ~/buildspace
wget http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz
tar xzf qhull-2020-src-8.0.2.tgz
cd qhull-2020.2

make
make install PREFIX=~/scilib
```

## Install GetFEM

```bash
cd ~/buildspace
wget http://download-mirror.savannah.gnu.org/releases/getfem/stable/getfem-5.4.1.tar.gz
tar xzf getfem-5.4.1.tar.gz
cd getfem-5.4.1/

./configure --with-pic --enable-python --enable-qhull --prefix=/home/<username>/scilib

make

make install 
```

\* Replace \<username> with your linux user name

## Copy GetFEM Python Interface

```bash
cp -r ~/scilib/lib/python3.8/site-packages/getfem ~/.conda/envs/metamtl-rl/lib/python3.8/site-packages
```

## Install Visualization Dependencies

```bash
pip install vtk==9.0.3 PyQt5 mayavi ipyevents
jupyter nbextension install --py mayavi --user
jupyter nbextension enable --py mayavi --user
```

## Install Visualization Package

```bash
conda install -c conda-forge pyvirtualdisplay
conda install -c conda-forge pyvista
conda install -c pyviz panel
```





## Possible Fix for GetFem Import Issue(Not tested yet)

Try this if the Python environment cannot find dependency of getfem after restarting the server on the cluster (this usually raises an error when import getfem in Python). The following code segment assume the environment already configured according to the previous sections.

This procedure will take about few minutes if the library was built successfully in the first time installation.

**This approach is not tested yet. Please provide feedback if anything not working**

```bash
conda activate metamtl-rl

export PATH=~/scilib/bin:$PATH
export C_INCLUDE_PATH=~/scilib/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/scilib/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/scilib/lib:$LIBRARY_PATH

cd ~/buildspace/qhull-2020.2
make
make install PREFIX=~/scilib

cd ~/buildspace/getfem-5.4.1
./configure --with-pic --enable-python --enable-qhull --prefix=/home/<username>/scilib
make
make install 

cp -r ~/scilib/lib/python3.8/site-packages/getfem ~/.conda/envs/metamtl-rl/lib/python3.8/site-packages
```

\* Replace \<username> with your linux user name






## set up env
Follow https://hoomd-blue.readthedocs.io/en/latest/installation.html 
or use the following script to build from source with MPI enabled.

```bash
mkdir workplace_compilation
cd workplace_compilation
# build a conda env
conda create -n osc_hpmc
conda install python==3.11.7

# check versions
gcc --version
mpirun --version

# get cmake
conda install cmake

# get source
wget https://github.com/glotzerlab/hoomd-blue/releases/download/v4.4.1/hoomd-4.4.1.tar.gz
tar -xvzf hoomd-4.4.1.tar.gz

# install dep
mv hoomd-4.4.1 hoomd-blue
python hoomd-blue/install-prereq-headers.py

# config and build
cmake -B build/hoomd -S hoomd-blue -DENABLE_MPI=ON
cmake --build build/hoomd # ~ 2hrs

# add to python path
cp -r hoomd-blue/hoomd ../
python_path_dir="$(dirname "$PWD")"
export PYTHONPATH="${PYTHONPATH}:${python_path_dir}"

# install other dep
pip install plato-draw
pip install gsd
pip install matplotlib
pip install shapely
pip install loguru
```


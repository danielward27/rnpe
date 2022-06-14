Package is rnpe. Scripts to recreate results in scripts.

Using conda, from the project root directory, the required python packages can be installed by running
```
conda env create -f environment.yml --name rnpe_env
```

The SIR example requires julia to be installed. The required julia packages can be downloaded using the Julia package REPL by running 
```
pkg> activate .
pkg> instantiate
```
Package to run the algorithm is in the rnpe folder. Scripts to recreate results are in the scripts folder. Using conda, from the project root directory, the required python packages can be installed by running

  conda env create -f environment.yml --name rnpe_env

The SIR example requires julia to be installed. The required julia packages can be downloaded using the Julia package REPL by running 

  | pkg> activate .
  | pkg> instantiate

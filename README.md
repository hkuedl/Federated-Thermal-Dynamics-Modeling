# Federated-Thermal-Dynamics-Modeling

_This work proposes a generalizable thermal dynamics modeling method by coordinating multiple buildings. It formulates a federated learning framework to facilitate collaborative modeling in a privacy-preserving way. It further identifies dual heterogeneity in both model structures and data distributions for federated learning, and then proposes a two-level personalization strategy combining similarity matrices and adaptive weighting to alleviate the impacts._

Codes for accepted Paper "Generalizable Thermal Dynamics Modeling via Personalized Federated Learning" in IEEE Transactions on Smart Grid.

Authors: Xueyuan Cui, Dalin Qin, Jean-François Toubeau, François Vallée, and Yi Wang.

## Requirements
Python version: 3.8.17

The must-have packages can be installed by running
```
pip install requirements.txt
```

## Experiments
### Data
All the data inputs for experiments can be acquired in ```Data``` here and ```Data_in``` that can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1CRcTHjpDVNd9OAcIsWxOow3TneJLg-6V?usp=sharing).

### Reproduction
To reproduce the experiments of the proposed methods and comparisons in the paper, please go to the folder
```
cd #Codes
```
where the introduction on the running order and each file's function is explained in ```readme.md```.

Note: There is NO multi-GPU/parallelling training in our codes. 

### Results
All models that are trained by the proposed method and other comparisons can be acquired in the folder ```#Results``` that can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/1CRcTHjpDVNd9OAcIsWxOow3TneJLg-6V?usp=sharing).

In particular, ```Models_newcomb```, ```Models_oneW```, ```Models_fedavg```, and ```Models_single``` includes models from "DW_ada", ("DW_fix"&"SW_data"&"SW_model"), "SW_avg", and "Local" methods, respectively.

## Citation
```
@ARTICLE{11071939,
  author={Cui, Xueyuan and Qin, Dalin and Toubeau, Jean-François and Vallèe, François and Wang, Yi},
  journal={IEEE Transactions on Smart Grid}, 
  title={Generalizable Thermal Dynamics Modeling via Personalized Federated Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Buildings;Data models;Load modeling;Adaptation models;Accuracy;Training;Collaboration;Mathematical models;Load forecasting;Federated learning;Building energy management;thermal dynamics;demand response;federated learning},
  doi={10.1109/TSG.2025.3585942}}
```
## Acknowledgments
Package ```#Codes/torchdiffeq1/``` is modified based on the open code of [Neural ODE](https://github.com/rtqichen/torchdiffeq). The rapid development of this work would not have been possible without this open-source package. 

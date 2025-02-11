# Federated-Thermal-Dynamics-Modeling

_This work proposes a generalizable thermal dynamics modeling method by coordinating multiple buildings. It formulates a federated learning framework to facilitate collaborative modeling in a privacy-preserving way. It further identifies dual heterogeneity in both model structures and data distributions for federated learning, and then proposes a two-level personalization strategy combining similarity matrices and adaptive weighting to alleviate the impacts._

Codes for submitted Paper "Generalizable Thermal Dynamics Modeling via Personalized Federated Learning".

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

The required models as the warm start of SMC are saved in ```#Results```.

## Citation
```
```

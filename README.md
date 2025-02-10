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
All the data for experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1wB3OkMHw7XF4DA5wYUdxXeCu_GbcM-Cv?usp=sharing).

### Reproduction
To reproduce the experiments of the proposed methods and comparisons for single-zone, 22-zone, and 90-zone buildings, please go to folders
```
cd #Codes/Single-zone
cd #Codes/22-zone
cd #Codes/90-zone
```
respectively. The introduction on the running order and each file's function is explained in ```Readme.md``` in the folder.

Note: There is NO multi-GPU/parallelling training in our codes. 

The required models as the warm start of SMC are saved in ```#Results```.

## Citation
```
```

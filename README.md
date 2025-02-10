# Federated-Thermal-Dynamics-Modeling

_This work innovates traditional model predictive control for the scheduling of thermostatically controlled loads. Inspired by "Smart Predict-then-Optimize", a "Smart Model-then-Control" strategy is proposed to learn a cost-effective model for the downstream control task. The actual control costs are reduced in multiple building types._

Codes for submitted Paper "A Smart Model-then-Control Strategy for the Scheduling of Thermostatically Controlled Loads".

Authors: Xueyuan Cui, Boyuan Liu, Yehui Li, and Yi Wang.

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

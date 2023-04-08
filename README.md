 
# Introduction

Code to the paper "GGT: Graph-Guided Testing for Adversarial Sample Detection of Deep Neural Network".

The implementation of this artifact is based on pytorch 1.6 with python 3.7. 

# Code Structure

This artifact includes four independent modules.

- model Generation (main)
- Adversarial Sample Generation (attacks)
- Label Change Rate and AUC over adversarial samples (lcr_auc)
- Adversarial Sample Detection (detect)


# Useage

### 1. Model Generation
Original model Generation:

```
python main.py --config ./config/CompeleteGraphMapResNet18.yml --multigpu 0
```
Graph-Guided model Generation 

```
python main.py --config ./config/OptimalGuassianMapResNet18.yml --multigpu 0 --search_direction min
```
### 2. Adversarial Samples Generation

fgsm attack:

```
python attacks/craft_adversarial_img.py --config ../config/CompeleteGraphMapResNet18.yml --multigpu 0 --pretrained [path of your original model] --attackType fgsm
```
### 3. Label Change Rate and AUC Calculation
mutated testing on normal sample:

```
python lcr_auc/mutated_testing.py --config ./config/CompeleteGraphMapResNet18.yml --multigpu 0 --prunedModelsPath [path of your Graph-Guided models] --testType normal > normal.log
```
lcr and auc calculation on normal sample:

```
python lcr_auc/lcr_auc_analysis.py --config ./config/CompeleteGraphMapResNet18.yml --multigpu 0 --isAdv False --maxModelsUsed 100 --lcrSavePath [path to save lcr result] --logPath [directory of your normal.log]
```
When finished, we can get lcr of normal samples(threshold for adversarial sample detection). The value is equal to: avg_lcr+99%confidence


### 4. Adversarial Sample Detection
Adversarial Samples detection:

```
python detect/adv_detect.py --config ./config/CompeleteGraphMapResNet18.yml --multigpu 0 --prunedModelsPath [path of your Graph-Guided models] --testSamplesPath [path of your adversarial samples] --threshold [lcr of normal samples] --testType adv
```

# Reference
- [dgl-prc/m_testing_adversatial_sample](https://github.com/dgl-prc/m_testing_adversatial_sample)













 

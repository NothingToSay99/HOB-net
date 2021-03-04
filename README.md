# HOB-net:High-order Block Network via Deep Metric Learning for Person Re-identification  

Implementation of the paper HOB-net:High-order Block Network via Deep Metric Learning for Person Re-identification.  
  
The approach proposed in the paper is simple yet effective on three re-ID datasets: Market-1501, DukdMTMC-ReID and CUHK03.  

![Alt text](https://github.com/NothingToSay99/HOB-net/blob/main/images/p2.png)

## Preparation

### 1.***Install***

　　pytorch>=**0.4** 

　　torchvision  

　　ignite=**0.1.2** 

　　yacs  

### 2.***Dataset***  

　　Download dataset to [reid/data/](https://github.com/NothingToSay99/HOB-net/tree/main/reid/data)

## Train  

## Demo

<img src="https://github.com/NothingToSay99/HOB-net/blob/main/images/dukeDemos.jpg" width="300" height="500" align="middle" alt="DukeMTMC"/>

## Results

| Dataset | Rank-1 | mAP |
| -- | :--: |:--:| 
| Market1501 | 94.7 | 86.3 |
| DukeMTMC | 88.2 | 77.2 |
| CUHK03-L | 69.4 | 66.5 |
| CUHK03-N | 69.2 | 66.8 |

## Acknowledgement

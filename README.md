# Future Frame Prediction 
Pytorch implementation of video anomaly detection paper for CVPR 2018: [Future Frame Prediction for Anomaly Detection – A New Baseline](https://arxiv.org/pdf/1712.09867.pdf).  
Most codes were obtained from the following GitHub page: [[Link]](https://github.com/feiyuhuahuo/Anomaly_Prediction)

I only trained the ```ped2``` dataset, the result:  

|     AUC                  |USCD Ped2    |
|:------------------------:|:-----------:|
| original implementation  |95.4%        |
|  this  implementation    |95.5%        |

### The network pipeline.  
![ffp_pipe](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/e9ce53b3-6c6f-453e-8b63-b05629c8b9d9)

## Environments  
PyTorch >= 1.1.  
Python >= 3.6.  
opencv  
sklearn  
Other common packages.  

## Prepare
- Download the ped2 dataset and put it under the ```data``` folder.

|     USCD Ped2            |
|:------------------------:|
| [Google Drive](https://drive.google.com/file/d/1lDhPPONJfivF_CtxIA3gg74f7RhNII-h/view?usp=drive_link)   | 

- Download the FlowNet2-SD weight and put it under the ```flownet/pretrained``` folder.

|     FlowNet2-SD.pth            |
|:------------------------:|
| [Google Drive](https://drive.google.com/file/d/1G3p84hzYRTCboNnJTb3iLwIPiHeNg-D_/view?usp=drive_link)   | 

- Download the trained model and put it under the ```weights``` folder.  

|     best_model_ped2.pth          |
|:------------------------:|
| [Google Drive](https://drive.google.com/file/d/1rHwcTnAcbEvHQb38dIK2FYzEleR6yP_a/view)   | 

## Train
```Shell
# default option.
python train.py --dataset=ped2 
# change 'seed'.
python train.py --dataset=ped2 --manualseed=50
# change 'max iteration'.
python train.py --dataset=ped2 --iters=60000
# change 'model save interval'.
python train.py --dataset=ped2 --save_interval=10000
# change 'validation interval'.
python train.py --dataset=ped2 --val_interval=1000
# Continue training with latest model
python train.py --dataset=ped2 --resume=latest_ped2
```
## Evalution
```Shell
# default option.
python eval.py --dataset=ped2 --trained_model=best_model_ped2
# change 'show heatmap'.
python eval.py --dataset=ped2 --trained_model=best_model_ped2 --show_heatmap=True
# change 'show roc_curve'.
python eval.py --dataset=ped2 --trained_model=best_model_ped2 --show_curve=True
```

## Results
#### Validation results can be found on the path ```results```by AUC graph.  
| AUC graph (ped2)                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![auc_graph](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/9c031b26-170a-4cf0-acee-bcd121e3f601) | 

#### Evaluation results can be found on the path ```results/ped2/{best_iter}``` by ROC Curve, Anomaly Score, etc.

| ROC Curve with AUC (ped2)                                                                               |
|----------------------------------------------------------------------------------------------------------------------|
|![toal_auc_curve](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/98443208-415d-4723-b78d-e8296766de32)| 

| Anomaly Score (ped2-04)                                                                               |
|----------------------------------------------------------------------------------------------------------------------|
|![score_graph](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/dfe6d801-112e-4189-b315-a8c3981c67b5) | 

| Frame Comparison (ped2-04)                                                                              |
|----------------------------------------------------------------------------------------------------------------------|
|![ffp_comparision](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/0c299ee4-7e40-495f-abf9-c7f7e11280cb) | 

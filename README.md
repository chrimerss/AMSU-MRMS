# Passive Microwave Precipitation Retrieval with deep learning

|Model description|inputs|learning type|epoches|loss|dice|threshold|name|
|:---------------:|:----:|:-----------:|:-----:|:--:|:--:|:-------:|:--:|
|LinkNet+ResNet18|amsu-a(1,2,3,4)+amsu-b(5 channels)|unfreeze|100|0.68|0.95|-4.464768/7|segmentation-class1|
|LinkNet+ResNet18|amsu-b (4channels)|unfreeze|100|0.60|0.82|0.75|Segmentation-4channels


In this study, we take two steps towards passive microwave (AMSU) precipitation retrival: first, segment satellite imagery into rain and no-rain classes (binary); second, apply second-round ML with rainy pixels.

## Pre-process

We crop AMSU satellite swath which is approximatly 45km at nadir into (64,64) sub-imageries randomly.

AMSU-A channels 1,2, 15 and AMSU-B channels 1, 2, 3, 4, 5 are selected as inputs because the low frequency channels (AMSU-A) are targeting water vapor in liquid phase, and higher frequencies (AMSU-B) are targeting mix-phase water.

As for the target, we mapped NSSL MRMS (multi-radar multi-sensor) ground based radar QPE to match the same spatiotemporary feature as AMSU flight. 

## Satellite imagery segmentation
In the imagery segmentation, we performed LinkNet with pretrained model that trained by imagenet.
<p align="center">
<img src='src/LinkNet-architecture.png'>

<p align="center"> Fig.1 LinkNet Architecture


### Comb1 - UNet + ResNet18 + 8 channels + 1 class 
__Loss__
<img src='src/LinkNetRes18-1class-8channels-loss.png'>

<p align="center"> Fig.2 Loss evolution with epoches

__Dice__
<img src='src/LinkNetRes18-1class-8channels-dice.png'>

<p align="center"> Fig.3 Dice evolution with epoches

__Results__

<p align="center"> 
<img src='src/LinkNetRes18-1class-8channels-kernelweights.png'>


<p align="center"> <img src='src/LinkNetRes18-1class-8channels-boxplot.png' width="50%">

<p align="center"> <img src="src/LinkNet-1class-8channels-confusionMatrix-test.png" width="50%"><img src="src/LinkNet-1class-8channels-confusionMatrix-val-benchmark.png" width="50%">

<p align="center"> Fig.4 LinkNet-1class-8channels-benchmark results 

<p align="center"> <img src='src/LinkNet-1class-8channels-PRAUC_curve.png' width="100%">

<p align="center"> Fig.5 PR-AUC curve to determine the best threshold 

<p align="center"> <img src='src/LinkNet-1class-8channels-optimalSurface.png' width="60%">

<p align="center"> Fig.6 objective surface plot.

## Rainfall retrieval

Attempt to use Random forest Regressor to quantify rain rate with grid search. The validation is based on KFolds, specifically 5 folds to validate data. It is running in 48 cores server, and it costs 60 hours to complete.

```python
# Grid search for hyperparameter tuning
rf= RandomForestRegressor()
hyperparam_grid= {
    'n_estimators': np.arange(10,500,20),
    'max_depth': np.arange(10,50,5),
    'warm_start':[True, False]
}
gridsearch= GridSearchCV(rf, hyperparam_grid, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
```

|Regressor|Parameters|median RMSE (benchmark)|model name|
|:-------:|:--------:|:--:|:--------:|
|Random Forest|depth-10,estimators-800|1.05(12.09)|model-1|

#### Results

<p align="center"><img src='src/LinkNet-ResNet-1class-8channels-rf-model1-spatial.png'>

<p align="center"> Fig.7 Spatial rainfall map for benchmark and model-1

<p align="center"><img src='src/LinkNet-ResNet-1class-8channels-rf-model1-rmse.png', width="50%">

<p align="center"> Fig.7 RMSE results for benchmark and model-1
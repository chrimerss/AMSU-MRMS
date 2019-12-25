# Passive Microwave Precipitation Retrieval with deep learning

|Model description|inputs|learning type|epoches|loss|dice|threshold|name|
|:---------------:|:----:|:-----------:|:-----:|:--:|:--:|:-------:|:--:|
|LinkNet+ResNet18|amsu-a(1,2,3,4)+amsu-b(5 channels)|unfreeze|100|0.68|0.78|0.9|segmentation-class1|


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
<img src='src/LinkNetRes18-1class-8channels-results.png'>

<img src='src/LinkNetRes18-1class-8channels-boxplot.png'>

<p align="center"> Fig.4 LinkNet-1class-8channels-benchmark results 

## Rainfall retrieval

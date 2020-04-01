# My solution for ICLR Workshop Challenge

Identify wheat rust in images from Ethiopia and Tanzania. The competition page can be found [here](https://zindi.africa/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease).

My Solution
- 5 fold cross validation
- Mixup
- EfficientNet Model trained on resized images 524x524 
- One Cycle Policy / Differential Learning rate using learning rate finder
- PseudoLabeling

My main misktakes
- Overfit to the leaderboard
- Not having a proper CV stratergy

```
Private Leaderboard rank 21.
Ideas should have tested
- Normalization Code (mean/std)
- TTA
- Gradient Clipping
- Remove confusing Images
- Builind a Widget to explore / remove data from the model
```

# NSFW, Score / Class, Model and Parameter Filter

This is a python script created to sort and filter images, mainly generated images from stable diffusion, based on their nsfw probability, score and class, model name and/or their parameters from generation.

## Requirements

- Python 3
- Windows 10 or Linux

## Installation
```bash
pip install pillow tensorflow keras PyQt5 opennsfw2
```

## Filtering types
There are currently 4 types of filtering available:
- NSFW probability
- Score
- Model (stable diffusion)
- Parameter(s) (positive prompt)

## NSFW probability

## Scoring models
There are several image scorer available:
- Xception
- VGG16
- VGG19
- ResNet50
- ResNet50V2
- ResNet101
- ResNet101V2
- ResNet152
- ResNet152V2
- InceptionV3
- InceptionResNetV2
- MobileNet
- MobileNetV2
- DenseNet121
- DenseNet169
- DenseNet201
- NASNetMobile
- NASNetLarge
- EfficientNetB0
- EfficientNetB1
- EfficientNetB2
- EfficientNetB3
- EfficientNetB4
- EfficientNetB5
- EfficientNetB6
- EfficientNetB7
- EfficientNetV2B0
- EfficientNetV2B1
- EfficientNetV2B2
- EfficientNetV2B3
- EfficientNetV2S
- EfficientNetV2M
- EfficientNetV2L
- ConvNeXtTiny
- ConvNeXtSmall
- ConvNeXtBase
- ConvNeXtLarge
- ConvNeXtXLarge

> More information about these can be found here: [Keras Applications](https://keras.io/api/applications/)

## Filter by stable diffusion model
If the images were generated from stable diffusion, you can sort them by their model name.

## Filter by stable diffusion positive input parameter(s)
If the images were generated from stable diffusion, you can filter and sort them by their parameter(s).
You can also input own parameters and only filter images that match your own parameter(s)
You can choose between matching only one or all of your parameter(s).

## ToDo
### Will be implemented in the future
- Add class names (print and option to name the folders)
- Parameters integrate with other (rewrite mode_folders)
- Better error handling and error messages
- Own ranges for nsfw and score (with more decimals, higher lows and/or higher highs)
- Automatic installation of necessary components
- Configuration file

### Will maybe be implemented

- Support for other image generation tools


## Help and bug reporting
Only use github issues for reporting bugs or feature requests 
Please report bugs only with detailed steps to reproduce.

## Credits

- Keras
- OpenNSWF2
- Stable Diffusion

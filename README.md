# Colorizing grayscale images

## WHAT?
Convert grayscale image to a colored image using different deep learning techniques.

## HOW?
We use 3 different models to try and colorize the grayscale image.
1. Deep Koalarization [1]
2. Inception-VGG AutoEncoder
3. VGG AutoEncoder
4. GAN (Experimental)

## SETUP
### DATA PREP
This project uses [Microsoft COCO Datset](http://cocodataset.org/#home)[2]. This project uses 2017 train, validation and test images. But, any year data should work if retraining the model.

Place the images in `./data/train`, `./data/validation` and `./data/test` folders.

### INSTALLING DEPENDENCIES
1. Install Python 3.6
2. Install `virtualenv`
3. Clone this repo
4. `cd` into the repo
5. Create a virtual environment
6. Run `pip install -r requirements.txt` (Use `requirements-gpu.txt` if using a GPU)

## LICENSE
This project is licensed under Apache License 2.0. The terms of the license can be found in [LICENSE](./LICENSE).

## REFERENCES
1. Baldassarre, Federico, Diego González Morín, and Lucas Rodés-Guirao. "Deep Koalarization: Image Colorization using CNNs and Inception-ResNet-v2." arXiv preprint arXiv:1712.03400 (2017).
2. Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014. 

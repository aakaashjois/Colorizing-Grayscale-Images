# Unsupervised Fully Automatic Colorization of Grayscale Images using Generative Adversarial Networks (UFACGIGAN)

## WHAT?
Converting grayscale images to color using a DCGAN

## HOW?
We model the generator based on the architecture reported by Iizuka et al [\[1\]](#references). For the discriminator, we use the VGG19 model with `categorical_cross_entropy` loss function. We train the generator on Microsoft's COCO dataset [\[2\]](#references).

## SETUP
### DATA PREP
This project uses [Microsoft COCO Datset](http://cocodataset.org/#home). This project uses 2017 train, validation and test images. But, any year data should work if retraining the model.

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
1. Iizuka, Satoshi, Edgar Simo-Serra, and Hiroshi Ishikawa. "Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification." ACM Transactions on Graphics (TOG) 35.4 (2016): 110.
2. Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.

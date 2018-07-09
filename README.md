# Differentiated Attentive Representation Learning for Sentence Classification
Implementation of the paper *Differentiated Attentive Representation Learning for Sentence Classification*.

## Environment
Tested on Python 2.7 and PyTorch 0.2.0.
[Tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) is needed if you want visualize the loss or accuracy of the model.

To run the DARLM, please first download some [raw data](https://drive.google.com/drive/folders/1KrfV8afg8LAhgXWewekM7cBT9IRmW-iW?usp=sharing) (corpus & wordvec), then put them in directory ``data/``.

## TREC dataset

### preprocessing

Use the following command to generate the data for model (or you can use above processed data):

``python data/trec_data/preprocess.py``


### training

To run the DARLM on TREC dataset, use:

``python train.py -config_file trec/hyper-param-trec.conf``

please specify which data file you will use in the config file.

## SST dataset

Refer to TREC.

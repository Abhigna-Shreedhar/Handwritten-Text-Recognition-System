# Handwritten Text Recognition with TensorFlow

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.

This Neural Network (NN) model recognizes the text contained in the images of segmented words.
As these word-images are smaller than images of complete text-lines, the NN can be kept small and training on the CPU is feasible.
3/4 of the words from the validation-set are correctly recognized and the character error rate is around 10%.

## Command line arguments

* `--train`: train the NN.
* `--validate`: validate the NN.
* `--beamsearch`: use vanilla beam search decoding (better, but slower) instead of best path decoding.
* `--wordbeamsearch`: use word beam search decoding (only outputs words contained in a dictionary) instead of best path decoding. This is a custom TF operation and must be compiled from source, more information see corresponding section below. It should **not** be used when training the NN.
* `--dump`: dumps the output of the NN to CSV file(s) saved in the `dump/` folder. 

If neither `--train` nor `--validate` is specified, the NN infers the text from the test image (`data/test.png`).


## Integrate word beam search decoding

Besides the two decoders shipped with TF, it is possible to use word beam search decoding.
Using this decoder, words are constrained to those contained in a dictionary, but arbitrary non-word character strings (numbers, punctuation marks) can still be recognized.


## Train model 

The data-loader expects the IAM dataset (or any other dataset that is compatible with it) in the `data/` directory.
If you want to train the model from scratch, delete the files contained in the `model/` directory.
Otherwise, the parameters are loaded from the last model-snapshot before training begins.
Then, go to the `src/` directory and execute `python main.py --train`.
After each epoch of training, validation is done on a validation set (the dataset is split into 95% of the samples used for training and 5% for validation as defined in the class `DataLoader`).
If you only want to do validation given a trained NN, execute `python main.py --validate`.

### Other datasets

Either convert your dataset to the IAM format or change the class `DataLoader` according to your dataset format.


## Information about model

### Overview

The implementation only depends on numpy, cv2 and tensorflow imports.
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
The illustration below gives an overview of the NN and here follows a short description:
* The input image is a gray-value image and has a size of 128x32
* 5 CNN layers map the input image to a feature sequence of size 32x256
* 2 LSTM layers with 256 units propagate information through the sequence and map the sequence to a matrix of size 32x80. Each matrix-element represents a score for one of the 80 characters at one of the 32 time-steps
* The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with best path decoding or beam search decoding (when inferring)
* Batch size is set to 50


### Analyze model

Run `python analyze.py` with the following arguments to analyze the image file `data/analyze.png` with the ground-truth text "are":

* `--relevance`: compute the pixel relevance for the correct prediction.
* `--invariance`: check if the model is invariant to horizontal translations of the text.
* No argument provided: show the results.

Results are shown in the plots below.
The pixel relevance (left) shows how a pixel influences the score for the correct class.
Red pixels vote for the correct class, while blue pixels vote against the correct class.
It can be seen that the white space above vertical lines in images is important for the classifier to decide against the "i" character with its superscript dot.
Draw a dot above the "a" (red region in plot) and you will get "aive" instead of "are".

The second plot (right) shows how the probability of the ground-truth text changes when the text is shifted to the right.
As can be seen, the model is not translation invariant, as all training images from IAM are left-aligned.
Adding data augmentation which uses random text-alignments can improve the translation invariance of the model.




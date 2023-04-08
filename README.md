# Dog Breed Classification on Stanford Dogs Dataset
## Description
The <a href= "http://vision.stanford.edu/aditya86/ImageNetDogs/">Stanford Dogs Dataset</a> contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.

### Pre-Requisites
For running the notebook on your local machine, following pre-requisites must be satisfied:
- Tensorflow 2.X
- Keras

## Approach
### Data Augmentation
Data augmentation is done through the following techniques:
- Rescaling to (256, 256)
- Dividing the data into train and validation by code

## Dividing the Dataset:

![image](https://user-images.githubusercontent.com/78736570/230719263-bc4ba370-e0f0-402f-928a-3660d5894273.png)


### Model Details
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 254, 254, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 127, 127, 16)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 32)      4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 62, 62, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 60, 60, 16)        4624      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 30, 30, 16)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 14400)             0         
                                                                 
 dense (Dense)               (None, 256)               3686656   
                                                                 
 dense_1 (Dense)             (None, 120)               30840     
                                                                 
=================================================================
Total params: 3,727,208
Trainable params: 3,727,208
Non-trainable params: 0
_________________________________________________________________

```

## Result

```

Epoch 1/20
452/452 [==============================] - 379s 833ms/step - loss: 21.0198 - accuracy: 0.0101 - val_loss: 4.7895 - val_accuracy: 0.0123
Epoch 2/20
452/452 [==============================] - 367s 812ms/step - loss: 5.7711 - accuracy: 0.0150 - val_loss: 4.7886 - val_accuracy: 0.0123
Epoch 3/20
452/452 [==============================] - 355s 786ms/step - loss: 4.8530 - accuracy: 0.0172 - val_loss: 4.7879 - val_accuracy: 0.0124
Epoch 4/20
452/452 [==============================] - 349s 773ms/step - loss: 4.7879 - accuracy: 0.0167 - val_loss: 4.7878 - val_accuracy: 0.0123
Epoch 5/20
452/452 [==============================] - 372s 824ms/step - loss: 4.8199 - accuracy: 0.0285 - val_loss: 4.7914 - val_accuracy: 0.0126
Epoch 6/20
452/452 [==============================] - 362s 800ms/step - loss: 4.7189 - accuracy: 0.0314 - val_loss: 4.8068 - val_accuracy: 0.0131
Epoch 7/20
452/452 [==============================] - 374s 827ms/step - loss: 4.6647 - accuracy: 0.0453 - val_loss: 4.8377 - val_accuracy: 0.0124
Epoch 8/20
452/452 [==============================] - 355s 786ms/step - loss: 4.7223 - accuracy: 0.0514 - val_loss: 4.8583 - val_accuracy: 0.0123
Epoch 9/20
452/452 [==============================] - 364s 805ms/step - loss: 4.5895 - accuracy: 0.0600 - val_loss: 4.9052 - val_accuracy: 0.0139
Epoch 10/20
452/452 [==============================] - 363s 802ms/step - loss: 4.5121 - accuracy: 0.0733 - val_loss: 4.9932 - val_accuracy: 0.0136
Epoch 11/20
452/452 [==============================] - 365s 807ms/step - loss: 4.5272 - accuracy: 0.0759 - val_loss: 5.0610 - val_accuracy: 0.0131
Epoch 12/20
452/452 [==============================] - 350s 775ms/step - loss: 4.4859 - accuracy: 0.0820 - val_loss: 5.1271 - val_accuracy: 0.0137
Epoch 13/20
452/452 [==============================] - 345s 764ms/step - loss: 4.4449 - accuracy: 0.0929 - val_loss: 5.2951 - val_accuracy: 0.0140
Epoch 14/20
452/452 [==============================] - 432s 956ms/step - loss: 4.4151 - accuracy: 0.1013 - val_loss: 5.5091 - val_accuracy: 0.0134
Epoch 15/20
452/452 [==============================] - 454s 1s/step - loss: 4.3596 - accuracy: 0.1127 - val_loss: 5.5994 - val_accuracy: 0.0134
Epoch 16/20
452/452 [==============================] - 473s 1s/step - loss: 4.2520 - accuracy: 0.1328 - val_loss: 5.6635 - val_accuracy: 0.0134
Epoch 17/20
452/452 [==============================] - 434s 960ms/step - loss: 4.2756 - accuracy: 0.1258 - val_loss: 5.8719 - val_accuracy: 0.0137
Epoch 18/20
452/452 [==============================] - 359s 795ms/step - loss: 4.1482 - accuracy: 0.1510 - val_loss: 6.0553 - val_accuracy: 0.0129
Epoch 19/20
452/452 [==============================] - 436s 964ms/step - loss: 4.0702 - accuracy: 0.1706 - val_loss: 6.1272 - val_accuracy: 0.0129
Epoch 20/20
452/452 [==============================] - 349s 773ms/step - loss: 4.0266 - accuracy: 0.1723 - val_loss: 6.3012 - val_accuracy: 0.0137
452/452 - 99s - loss: 4.3879 - accuracy: 0.1004 - 99s/epoch - 220ms/step
192/192 - 42s - loss: 6.3012 - accuracy: 0.0137 - 42s/epoch - 218ms/step
Training loss: 4.387875080108643
Training accuracy: 0.10035966336727142
Validation loss: 6.301185607910156
Validation accuracy: 0.013721005991101265

```

## References
- The original data source is found on http://vision.stanford.edu/aditya86/ImageNetDogs/ and contains additional information on the train/test splits and baseline results.

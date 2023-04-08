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

##Dividing the Dataset:

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

## References
- The original data source is found on http://vision.stanford.edu/aditya86/ImageNetDogs/ and contains additional information on the train/test splits and baseline results.

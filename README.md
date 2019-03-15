# Breast-Area-Sgementation
Breast are segmentation example
==================================
U-nets
---------------------------------
# Database
The databse are from the GWU hospital, it contains 11 patiens, 165 infrared images.
# Enviroment
We run the U-Nets model under Keras 2.2.2.
# Usage:
## Data prepare:
Run the load_data.py. Chnage the data paths get the train data and test data in npy form.
## Training
Run the Model.py. The net work are training. After training, the model are saved automatically. 
## Testing
Run the test.py. 

Image Transformation
------------------------
Run the image_transformation.m, the test image is in the folder called trans.

CLAHE
-------------------------
Run the CLAHE.m, the test image is in the folder called trans.

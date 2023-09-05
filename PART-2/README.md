# Camera Trap Application
In this section, we'll build an image classification model to classify between Impalas, Warthogs and Zebras in the [DeKUT Conservancy](https://conservancy.dkut.ac.ke/).
This model is intended for deployment on the OpenMV Cam H7 device. When the model is used for inference, it will visually indicate different animal classes by displaying distinct colors.

## Getting you started
```
git clone https://github.com/DeKUT-DSAIL/dsail-tech4wildlife.git
cd dsail-tech4wildlife
```

### Install and create a virtual environment
```
pip3 install virtualenv
virtualenv tech4wildlife
```
### Activating the virtual environment<br>

#### Linux       
```
source tech4wildlife/bin/activate
```
#### Windows    
```
.\tech4wildlife\Scripts\activate
```

### Install the required dependencies
```
pip install -r requirements.txt
```

# DSAIL Tech4Wildlife Workshop
This repository contains content to be delivered during the DSAIL Tech4Wildlife Workshop in collaboration with [Fauna \& Flora](https://www.fauna-flora.org/).  During this workshop we will demonstrate how AI can be used to aid conservation efforts. In particular we will build a camera trap capable of species identification at the Edge!

## About DSAIL

## About F\&F


## Course Topics
### PART 1 (Introduction and building a baseline model)
- Hardware description
- Hardware and Software Setup
- Image Classification (Bottles and Computer screens)
- Object Detection (Face detection and bottle detector)

### PART 2 (Camera Trap Application)
- Image Classification

### Requirements
- Arduino Tiny Machine Learning Kit
- Open MV Cam H7
- An account with Edge Impulse
- Installed Arduino CLI package
- Installed Edge Impulse CLI
- This cloned repository

## Hardware Description

### Arduino Nano 33 BLE Sense
The Arduino Nano 33 BLE Sense is a compact microcontroller board that features the nRF52840 from Nordic Semiconductors, a 32-bit ARM® Cortex®-M4 CPU running at 64 MHz and a built-in Bluetooth Low-Energy (BLE) module.

![image](https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/52fce3fc-5e93-4a5e-84b2-47bb4e48e18b)

### OV7675 Camera Module
The OV7675 camera module is a small, low-cost camera sensor that can capture images and video. It has a resolution of 640x480 pixels and communicates with the Arduino Nano through the I2C interface.

![image](https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/2ca87437-0aa2-416d-890c-a0e897ae9a2f)

### Tiny Machine Learning(TinyML Shield)
The shield includes all the necessary circuitry to connect the camera module to the Arduino Nano, eliminating the need for manual wire connections. It also ensures the correct pinout, making it easier for us to set up and work with the hardware.

![image](https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/81b86ef2-b6ea-40f7-aa0c-8c128cf4b7bb)


### Open MV Cam H7
This is a compact embedded vision camera module featuring a powerful ARM microcontroller with a Cortex-M7 core running at 480 MHz. 
This high clock speed enables efficient and speedy inferencing, making it ideal for real-time computer vision applications. 

![image](https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/64513ee4-928f-4beb-b51a-e5ba95baf528)

## Software Setup 
- Start by setting up an Edge Impulse Account [here](https://studio.edgeimpulse.com/signup)
- Install the Edge Impulse CLI [here](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation)
- Install the Arduino CLI [here](https://arduino.github.io/arduino-cli/0.23/installation/)
### Steps to setting up this repository

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
cd PART-2
pip install -r requirements.txt
```
This will get you started with everything required for this tutorial.<br>
For a step-by-step guide on developing your baseline model and applying that knowledge to a camera trap application.<br>
Follow the instructions highlighted [here](https://dekut-dsail.github.io/tutorials/image_classification.html)
















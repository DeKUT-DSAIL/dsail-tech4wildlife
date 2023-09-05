## A Baseline Model
In this section, we'll build a baseline model that will serve as a foundational reference point for evaluating the performance of more complex models for the next step in a camera trap application.
We will explore image classification and object detection using the Edge Impulse interface, allowing you to train and deploy models capable of recognizing objects such as bottles and computer screens.
We will utilize the TinyML kit for image classification and OpenMV for object detection.

### Outline
- Image Classification (Bottles and Computer screens)
- Object Detection (Face detection and bottle detector)

## Software Setup 
- Start by setting up an Edge Impulse Account [here](https://studio.edgeimpulse.com/signup)
- Install the Edge Impulse CLI [here](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation)
- Install the Arduino CLI [here](https://arduino.github.io/arduino-cli/0.23/installation/)
- Install the OpenMV IDE [here](https://openmv.io/pages/download)
  
### Steps to setting up this repository

```
git clone https://github.com/DeKUT-DSAIL/dsail-tech4wildlife.git
cd dsail-tech4wildlife
```

This will get you started with everything required for this tutorial.<br>
For a step-by-step guide on developing your baseline model, follow the instructions highlighted [here](https://dekut-dsail.github.io/tutorials/image_classification.html)
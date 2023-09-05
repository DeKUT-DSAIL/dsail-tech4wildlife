# DSAIL Tech4Wildlife Workshop
This repository contains content to be delivered during the DSAIL Tech4Wildlife Workshop in collaboration with [Fauna \& Flora](https://www.fauna-flora.org/).  During this workshop, we will demonstrate how AI can be used to aid conservation efforts. In particular, we will build a camera trap capable of species identification at the Edge!

## About DSAIL
Centre for Data Science Artificial Intelligence(DSAIL) envisions itself as a preeminent centre at the forefront of data science and artificial intelligence, setting new standards in research and problem-solving. With a dedicated mission to cultivate an environment conducive to researchers and practitioners in these fields, DSAIL is committed to nurturing innovative solutions tailored to the unique challenges of its local context. The lab's strategic approach centres on the design, creation, and development of cutting-edge technological solutions, poised to address and resolve real-world problems, making a profound impact on both academia and society at large.

## About F\&F
Fauna & Flora, an international wildlife conservation charity with a legacy spanning over 120 years, is dedicated to safeguarding the world's natural wonders. With a global team spread across the planet, they collaborate closely with conservation partners in over 40 countries to preserve and rejuvenate habitats, rescue endangered species from the brink of extinction, and empower local communities to embrace sustainable livelihoods harmoniously with nature. From their UK headquarters and regional offices worldwide, they blend collective wisdom and technical prowess to tackle multifaceted challenges, encompassing issues like habitat degradation, illegal wildlife trade, climate change, plastic pollution, corporate sustainability, and global policy, with an unwavering commitment to nurturing biodiversity across the globe.


## Course Topics
### PART 1 (Introduction and building a baseline model)
- Hardware description
- Hardware and Software Setup
- Image Classification (Bottles and Computer screens)
- Object Detection (Face detection and bottle detector)

### PART 2 (Camera Trap Application)
- Image Classification (Impala, Warthog and Zebra)

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

<div align="center">
  <img src="https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/eac1411e-d0bf-46f8-919a-a336aab558bd" alt="Arduino Nano 33 BLE Sense">
</div>


### OV7675 Camera Module
The OV7675 camera module is a small, low-cost camera sensor that can capture images and video. It has a resolution of 640x480 pixels and communicates with the Arduino Nano through the I2C interface.


<div align="center">
  <img src="https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/7b1cd7b1-3a8a-4e45-916e-893872e23c3b" alt="OV7675 Camera Module">
</div>


### Tiny Machine Learning(TinyML Shield)
The shield includes all the necessary circuitry to connect the camera module to the Arduino Nano, eliminating the need for manual wire connections. It also ensures the correct pinout, making it easier for us to set up and work with the hardware.

<div align="center">
  <img src="https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/a0656d21-77d4-4227-985d-a0caa061dd40" alt="Tiny Machine Learning(TinyML Shield)">
</div>



### Open MV Cam H7
This is a compact embedded vision camera module featuring a powerful ARM microcontroller with a Cortex-M7 core running at 480 MHz. 
This high clock speed enables efficient and speedy inferencing, making it ideal for real-time computer vision applications. 

<div align="center">
  <img src="https://github.com/DeKUT-DSAIL/dsail-tech4wildlife/assets/88529649/b7349310-5c72-4cce-851c-e50bfe9bb200" alt="Open MV Cam H7">
</div>

## Software Setup 
- Start by setting up an Edge Impulse Account [here](https://studio.edgeimpulse.com/signup)
- Install the Edge Impulse CLI [here](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation)
- Install the Arduino CLI [here](https://arduino.github.io/arduino-cli/0.23/installation/)
### Steps to setting up this repository

```
git clone https://github.com/DeKUT-DSAIL/dsail-tech4wildlife.git
cd dsail-tech4wildlife
```

This will get you started with everything required for this tutorial.<br>
For a step-by-step guide on developing your baseline model and applying that knowledge to a camera trap application.<br>
Follow the instructions highlighted [here](https://dekut-dsail.github.io/tutorials/image_classification.html)
















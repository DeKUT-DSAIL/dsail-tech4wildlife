# DSAIL Tech4Wildlife Workshop
This repository contains content to be delivered during the DSAIL Tech4Wildlife Workshop in collaboration with [Fauna \& Flora](https://www.fauna-flora.org/).  During this workshop we will demonstrate how AI can be used to aid conservation efforts. In particular we will build a camera trap capable of species identification at the Edge!

## About DSAIL

## About F\&F


## Course Topics
## Arduino Nano BLE 33 Sense and OV7670

This project aims to deploy a pre-trained image classifier on an Arduino Nano 33 BLE Sense using the OV7670 camera module. The inferenced results will be displayed on a Flask app running on a computer connected to the Arduino via Bluetooth.


## Steps

1. **TensorFlow Model Preparation:**
   - Train and optimize an image classifier model using TensorFlow on a computer.
   - Ensure that the model is compatible with TensorFlow Lite.

2. **TensorFlow Lite Conversion:**
   - Convert the trained TensorFlow model to the TensorFlow Lite format.
   - Use the TensorFlow Lite Converter for the conversion process.

3. **Arduino Setup:**
   - Wire and configure the OV7670 camera module with the Arduino Nano 33 BLE Sense.
   - Refer to the Arduino documentation or online resources for detailed instructions.

4. **TensorFlow Lite on Arduino:**
   - Install the TensorFlow Lite library in the Arduino IDE.
   - Use the library to enable TensorFlow Lite model inference on the Arduino Nano 33 BLE Sense.

5. **Model Deployment:**
   - Upload the converted TensorFlow Lite model to the Arduino Nano 33 BLE Sense.
   - Utilize the Arduino IDE or Arduino CLI for the model deployment.

6. **Bluetooth Communication:**
   - Implement Bluetooth communication between the Arduino Nano 33 BLE Sense and the computer running the Flask app.
   - Use the ArduinoBLE library on the Arduino side and a Bluetooth library/module on the Flask app side.

7. **Flask App:**
   - Develop a Flask app on the computer to receive data from the Arduino Nano 33 BLE Sense over Bluetooth.
   - Display the inferenced results from the image classifier on the Flask app.

8. **Integration:**
   - Integrate the TensorFlow Lite inference code with the Arduino code.
   - Capture images from the OV7670 camera module, perform inference using the deployed model, and send the results to the Flask app over Bluetooth.


## License

This project is licensed under the [MIT License](LICENSE).







## Hardware Description

# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, uos, gc, pyb

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)      # Set frame size to QVGA (320x240)
#sensor.set_windowing((256, 256))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
print('Begin')
#print(gc.mem_free()/1024)
##print(help(tf.load))
#net = tf.load("trained6.tflite", load_to_fb=uos.stat('trained6.tflite')[6] > (gc.mem_free() - (64*1024)))
#print('Loaded')
#print(gc.mem_free())
# Initialize LEDs
led_red = pyb.LED(1)  # Red LED
led_green = pyb.LED(2)  # Green LED
led_blue = pyb.LED(3)  # Blue LED

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("trained445.tflite", load_to_fb=uos.stat('trained445.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained445.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        #print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        #img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))
        pred_dict = dict(predictions_list)
        highest_class = max(pred_dict, key=pred_dict.get)# Control LEDs based on the highest predicted class
        print(highest_class, pred_dict[highest_class])
        if highest_class == "BUSHBUCK":# red
            led_red.on()
            led_green.off()
            led_blue.off()
        elif highest_class == "IMPALA": # green
            led_red.off()
            led_green.on()
            led_blue.off()
        elif highest_class == "MONKEY":# blue
            led_red.off()
            led_green.off()
            led_blue.on()
        elif highest_class == "WARTHOG":# yellow
            led_red.on()
            led_green.on()
            led_blue.off()
        elif highest_class == "WATERBUCK":# white
            led_red.on()
            led_green.on()
            led_blue.on()
        elif highest_class == "ZEBRA":# purple
            led_red.on()
            led_green.off()
            led_blue.on()
            
        #for i in range(len(predictions_list)):
            ##print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))
            #print(highest_class)

    print(clock.fps(), "fps")

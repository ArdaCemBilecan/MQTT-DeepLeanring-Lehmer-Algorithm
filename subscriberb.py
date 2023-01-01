import numpy as np
import json
import paho.mqtt.client as mqtt
import base64
import time
import matplotlib.pyplot as plt
import cv2


def on_message(client,userdata,message):
    print("Image Received")
    
    m_decode=str(message.payload.decode("utf-8","ignore"))
    dic = json.loads(m_decode)
    img = base64.b64decode(dic['Image'])
    key = base64.b64decode(dic['Key'])
    name = dic['Name']

    # path_name = name.decode("utf-8") 
    # print(path_name)
    # print(type(path_name))
    
    np_img = np.frombuffer(img, dtype=np.uint8)
    np_key = np.frombuffer(key, dtype=np.uint8)
    
    img_cv = cv2.imdecode(np_img,cv2.IMREAD_COLOR)
    key_cv = cv2.imdecode(np_key,cv2.IMREAD_COLOR)
    
    
    
    decrypted_image = np.zeros((256, 256, 3),dtype='uint8')
    for row in range(256):
        for column in range(256):
            for depth in range(3):
                decrypted_image[row, column, depth] = img_cv[row, column, depth] ^ key_cv[row, column, depth]
                
    
    plt.imsave('C:/Users/Asus/Desktop/Lehmer/MQTT/subs/{name}'.format(name = name),decrypted_image)
    
    
    


mqttBroker='mqtt.eclipseprojects.io'
client = mqtt.Client('Image')
client.connect(mqttBroker)
client.subscribe('Sent_Images2')

client.loop_start()
while True:
    client.on_message = on_message
    
time.sleep(50)

client.loop_stop()
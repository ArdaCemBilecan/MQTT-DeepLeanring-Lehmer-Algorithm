import base64
from glob import glob
import cv2
import json
import paho.mqtt.client as mqtt
import time
import Lehmer
import numpy as np
from sklearn.preprocessing import minmax_scale
from CLIP import Clip


class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)

def normalize(volume):
    shape = volume.shape
    lvolume = list(volume.ravel())
    norm_list = minmax_scale(lvolume)
    rvolume = np.array(norm_list).reshape(shape) * 255
    rvolume = np.asarray(rvolume,dtype='uint8')
    return rvolume
    
def encryption(img,key):
    encrypted_image = np.zeros((256, 256, 3),dtype='uint8')
    for row in range(256):
        for column in range(256):
            for depth in range(3):
                encrypted_image[row, column, depth] = img[row, column, depth] ^ key[row, column, depth]
    
    return encrypted_image

def validation(img):
    
    labels = Clip.get_output(img)
    return Clip.find_max(labels)


mqttBroker='mqtt.eclipseprojects.io'
client = mqtt.Client('Sent')
client.connect(mqttBroker)

img_paths = glob('test/*')
lehmer = Lehmer.Lehmer(10000,100000,15,0)
messages= []

print('For started')

i = 0
for path in img_paths:
    print(i)
    i += 1
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
    
    img_path_encode = path.split('\\')[1]
    # img_path_encode = img_path_encode.split('.')[0]
    img_path_encode = img_path_encode.encode('utf-8')
        
    
    key = lehmer.generate_key()
    normalize_key = normalize(key)
    
    encryption_img = encryption(img,normalize_key)
    
    _, buffer = cv2.imencode('.png', encryption_img)
    png_as_txt = base64.b64encode(buffer)
    
    
    _,buffer_key = cv2.imencode('.png', normalize_key)
    key_as_txt = base64.b64encode(buffer_key)
    
    message = json.dumps({'Image':png_as_txt , 
                          'Key': key_as_txt,
                          'Name':img_path_encode,
                          'Label':validation(img)[0],
                          'Value':validation(img)[1]},cls = BytesEncoder)
    #print(message)
    messages.append(message)
    


print('For Ended')
print('Sending will start in 10 ms')
time.sleep(10)
i = 0
while i < len(messages):
    client.loop_start()
    client.publish('Sent_Images2',messages[i])
    client.loop_stop()
    print('Just Published Images')
    i=i+1
    time.sleep(15)
    
    
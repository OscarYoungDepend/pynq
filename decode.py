#!/usr/bin/env python
# encoding: utf-8
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import glob
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt

#picture_dir=os.path.join(os.getcwd(),'*.jpg')
#for jpgfile in glob.glob(picture_dir):
    #encode()
    #file = open(jpgfile,'rb')
    #base64_data = base64.b64encode(file.read())
    #decode()
    # byte_data = base64.b64decode(base64_data)
    # image_data = BytesIO(byte_data)
    # img = Image.open(image_data)
    # #show pioture
    # plt.imshow(img)
    # plt.show()
def Decode():
    with open('/home/xilinx/jupyter_notebooks/spaq/picture.txt','r',encoding='utf-8') as f:
        data = f.read()
        print(type(data))
        missing_padding = 4 - len(data) % 4
        if missing_padding:
            data += '='* missing_padding
        img = Image.open(BytesIO(base64.b64decode(data)))
        img.save('/home/xilinx/jupyter_notebooks/spaq/images/picture.jpg')
        print("图片成功转码并保存")

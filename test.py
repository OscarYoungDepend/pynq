# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:36:54 2020

@author: mi
"""


import my
import decode
import serial
ser=serial.Serial("/dev/ttyUSB0",115200,timeout=10)
#print(ser)
print("串口连接成功，串口信息：\n")
print(ser)
while(1):
    txt=ser.readlines(40000)
    if  txt!=[]:
        if len(str(txt[0]))>4000:
            print("成功接收蓝牙数据")
            with open("/home/xilinx/jupyter_notebooks/spaq/picture.txt", "w", encoding='utf-8') as f:
                f.write(str(txt[0])[2:-2])
                print("成功保存图片base64编码数据")
                f.close()
            break

decode.Decode()
score=my.test()
ser.write(score.encode("utf-8"))
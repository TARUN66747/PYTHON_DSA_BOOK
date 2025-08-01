# -*- coding: utf-8 -*-
"""
Created on Tue May 13 22:19:05 2025

@author: harsh
"""
'''
seek():
The seek() function in Python is used to change the 
position of the file pointer within a file. 
It allows you to move to a specific location in 
the file to read or write data    

tell():
    The tell() method in Python is used to determine 
    the current position of the file pointer within a 
    file. It returns the position as an integer, 
    representing the number of bytes from the 
    beginning of the file
'''
'''
with open('naruto.txt','r') as f:
    print(type(f))
    
    f.seek(10)#reads after 10 character
    print(f.tell())
    data =f.read(5)
    print(data)
    print(f.tell())#used to tell position of letter
'''
with open('sample.txt','w') as k:
    k.write("hello world")
    k.truncate(5)#to determine file bytes and limit


with open('sample.txt','r') as k:
    print(k.read())
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:23:51 2025

@author: harsh
"""

class Employee:
    def __init__(self,name,_id):
        self.name = name
        self.id = _id 
        
        
class programmer(Employee):
    def __init__(self,name,_id,lang):
        super( ). __init__(name,_id)
        self.lang = lang
        
rohan = Employee("rohandas","420")
tarun =programmer("tarun","8656","python")
print(rohan.name)
print(tarun.name,tarun.id,tarun.lang)
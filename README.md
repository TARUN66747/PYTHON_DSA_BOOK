# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:20:58 2025

@author: harsh
"""
"""
seq=[1,2,3,4,5,6,7,8,9]
print(seq[::2])
print(seq[::-1])
for i,value in enumerate(seq):
    print(i,"+",value,i+value)
    
words=["apple","bat","bar","atom","book"]
by_letter={}

for i in words:
    letter=i[0]
    if letter not in by_letter:
        by_letter[letter]=[i]
    else:
        by_letter[letter].append(i)
        
print(by_letter)
print(hash("string"))
print(hash([1,2]))
"""
"""
all_data=[["john","emily","micheal","mary","steven"],["maria","juan","xavier","nataliya","pilar"]]
names_of_intrest=[]

for names in all_data:
    enough_es=[name for name in names if name.count("e")>=2 ]
    names_of_intrest.extend(enough_es)
    
print(names_of_intrest)
"""
"""
result=[name for names in all_data for name in names if name.count("e") >=2]
print(result)
"""
"""
states=["albama","Georgia!","georgia","Flo@rida","south 5ca#rolina","west virginia?"]
import re
def clean_strings(k):
    result=[]
    for value in k:
        
        value=value.strip()
        value=re.sub("[!#?@5]","",value)
        value=value.title()
        result.append(value)
    return result

print(clean_strings(states))
"""

"""
def apply_to_list(some_list,f):
    return [f(x) for x in some_list ]

ints=[1,2,3,4,5,6]
y=lambda x:x+2

print(apply_to_list(ints,y))
print(apply_to_list(ints,lambda x:x**2))
"""
"""
strings=["foo","card","bar","aaaa","abab"]
strings.sort(key=lambda x:len(set(list(x))))
print(strings)
"""
"""
def add_numbers(x,y):
    return x+y;
from functools import partial
add_five= partial(add_numbers,5)
"""
"""
dicta={"a":1,"b":2,"c":3}
for key in dicta:
    print(key)
dict_iterator=iter(dicta)
print(dict_iterator)
"""
"""
def squares(n=10):
    print(f"generating squares from 1 to {n}")
    for i in range (1,n+1):
        yield i**2
print(squares())
"""
"""
def my_generator():
    for i in range(20):
        yield i
         
gen=my_generator()
print(next(gen))

for j in gen:
    print(j)
    """
"""
gen=(x**2 for x in range(100))
"""
"""
def _make_gen():
    for x in range(100):
        yield x**2
gen = _make_gen()

print(gen)
        
print(sum(x**2 for x in range(100)))
print(dict((i,i**2) for i in range(5)))
"""
"""
import itertools
first_letter=lambda x:x[0]
names=["alan","adam","wes","will","albert"]
for letter,names in itertools.groupby(names,first_letter):
    print(letter,list(names))
"""
"""
import itertools
first_letter=lambda x:x[0]
names=["alan","adam","wes","will","albert"]
for letter,names in itertools.combinations(names,first_letter):
    print(letter,list(names))
"""
"""
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x
print(attempt_float(("tarun")))
"""
"""
def attempt_float(x):
    try:
        
        return float(x)
    except(TypeError,ValueError):
        return x
print(attempt_float(("tarun")))
"""
"""
names=["alan","adam","wes","will","albert"]
first_letter=map(lambda x:x[0],names)

for Fletter,name in zip(first_letter,names):
    print(Fletter,name)
"""
"""
path=""
f=open(path)
print(f.read(200))
f.close()
f2=open(path,"rb")
print(f2.read(200))
print(f2.tell()) 
"""
"""
import sys
print(sys.getdefaultencoding())
path=r"C:/Users/harsh/OneDrive/Desktop/tarun work/tarun work/javascript start.txt"
f= open(path)
print(f.seek(3))
print(f.read(1))
print(f.read(2))
f.close()
path2 = r"C:/Users/harsh/OneDrive/Desktop/tarun work/tarun work/demon.txt"
def kchecker(x):
    z=0
    for i in x:
        if i=="k":
            z=z+1
    if z > 3:
        return 1
    else:
        return 0
     
        
with open(path2,"w") as k:
    print(k.writelines(x for x in open(path) if kchecker(x)==1))

"""
"""
path=r"C:/Users/harsh/OneDrive/Desktop/tarun work/tarun work/javascript start.txt"
with open(path , "rb") as f:
    chars = f.read(255)
    print(chars)
print("************************")
chars.decode("utf8")
print(chars.decode("utf8"))
"""
"""
import numpy as np

array=np.arange(10000000)
lista=list(range(10000000))
array2=()
"""
"""
import numpy as np
#data = np.random.randn(2, 3,4,5)
#print(data)
#print("************************************************************")
#print(data*10)
data=[1,2,3,4,5]
k = np.array(data)
print(k)
print(type(k))
data2=[[1,2,3,4,5],[1,2,45,6,7],[8,9,10,11,12]]
array2=np.array(data2)
print(array2)
print(array2.shape)
kata=np.zeros(10)
bata=np.array(kata)
print(bata)
kata2=np.zeros((8,8))
bata2=np.array(kata2)
print(bata2)
print( type(bata2))
kata3=np.empty((5,5))
bata3=np.array(kata3)
print(bata3)
"""
"""
import numpy as np
bata = np.arange(1,10)
print(bata)
print (type(bata))
lauki=np.zeros((4,4))
kaadu=np.ones_like(lauki)
print(kaadu)
print(np.ones((2,3)))
print(np.identity((4)))
print(np.eye((6)))
"""
"""
t=6
a=0
l=0
for i in range(5):
    print("*"*t," "*a," "*l,"*"*t ,sep="")
    t=t-1
    a=a+1
    l=l+1
    if t==1:
        for i in range(20):
           
            t=t+1
            a=a-1
            l=l-1
            print("*"*t," "*a," "*l,"*"*t,sep="")
    
            if a==0 :
                break 
"""
"""

star=5
space=0
message=0
for i in range(9):
    print("*"*star+"  "*space+"*"*star)
    if star>1 and message==0:
        star-=1
        space+=1
    else:
        message=1
        star+=1
        space-=1
"""
"""
import numpy as np
names = np.array(["bob","joe","will","bob","will","joe","joe"])
data = np.random.randn(7,4)
print(data)
print("*******************************")
print(names=="bob")
print(data[names=="bob",2:])
print("*******************************")
print(names != "bob")
print(~(names == "bob"))
print("*******************************")
cond = names =="bob"
print(data[~cond])
print("*******************************")
data[data<0]=0
print(data)
print("*******************************")
data[names!="joe"]=7
print(data)
print("*******************************")
"""
"""
import numpy as np
arr = np.empty((8,4))
for i in range (8):
    arr[i]=i
    
#print(arr)
#print(arr[[4,3,0,6]])
print(arr[-3,-5,-7])
"""
"""
import numpy as np
arr=np.arange(32).reshape((8,4))
#print(arr)

#print(arr[[1,5,7,2],[0,3,1,2]])
print(arr[[1,5,7,2]][:,[0,3,1,2]])
"""
"""
import numpy as np
arr = np.arange(16).reshape((2,2,4))
print(arr)
print("**********************************************")
print(arr.transpose((1,0,2)))
print("**********************************************")
print(arr.T)
print("**********************************************")
print(arr.swapaxes(1,2))
"""
"""
import numpy as np
arr = np.array((4,9,16,25))
print(np.sqrt(arr))
"""
"""
import numpy as np
x = np.random.randn(8)
y =np.random.randn(8)
print(x)
print(y)
print(np.maximum(x,y))
"""
"""
import numpy as np
points = np.arange(-5,5,0.01)
#print(points)
xs,ys= np.meshgrid(points,points)
#print(ys)
#print(xs)
z = np.sqrt(xs**2+ys**2)
#print(z)


import matplotlib.pyplot as plt
print(plt.imshow(z,cmap=plt.cm.gray))
"""
"""
xarr=np.arange(1.1,1.5,0.1)
yarr = np.arange(2.1,2.5,0.1)
cond= np.array([True,False,True,True,False])
result =[(x if c else y) for x,y,c in zip(xarr,yarr,cond)]
print(result )
"""
"""
import numpy as np
arr1 = np.random.randn(5,4)
#print(arr1.mean())
#print(np.mean(arr1))
#print(np.sum(arr1))
#print(arr1) 
print("************************************************************")
#print(arr1.mean(axis=1))
print("************************************************************")
print(arr1.mean(1))
#print(arr1.sum(1))
"""
"""
import numpy as np
arr5 = np.random.randn(2,2)
print(arr5)
#print(arr5.cumsum(1))
#print(arr5.cumprod(1))
print(np.min(arr5))
print(np.max(arr5))
"""
"""
import numpy as np
arr = np.random.randn(100)
print((arr>1).sum())
"""
"""
import numpy as np
arr =np.random.randn(3,3)
print(arr)

for i in range(3):
    for j in range(3):
        if arr[i][j]>0:
            arr[i][j]= 1
        else:
            arr[i][j] = 0
print(arr)
print(arr.any())
print(arr.all())
arr2 = np.ones((3,3))

print(arr2.all())
"""
"""
import numpy as np
l_arr =  np.random.randn(1000)
l_arr.sort()
print(l_arr[int(0.05*len(l_arr))])
"""
"""
import numpy  as np
arr = np.array(["bob","joe","will","bob","will","joe","joe"])
print(np.unique(arr))
values= np.array([6,0,0,3,2,5,6])
print(np.in1d(values,[2,3,6]))
"""
"""
import numpy as np
art = np.arange(1,15) 
np.save("satvik_1",art)
print(np.load("satvik_1.npy"))
a2 = np.ones(3)
print(a2)
"""
"""
import numpy as np
from numpy.linalg import inv,qr
X = np.random.randn(5,5)
mat = X.T.dot(X)
print(inv(mat))
print(mat.dot(inv(mat)))
print("*********************************************************************")
q,r = qr(mat)
print(r)
print("********************************************************************")
print(q)
t=q.T
print(q.dot(t))
"""
import numpy as np
samples = np.random.normal(size=(4,4))
print(samples)
from random import normalvariate
N = 1000000



    
    

    

   
    


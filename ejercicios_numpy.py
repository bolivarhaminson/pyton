from pandas import Series
import random


obj = Series([0] * 10)

for i in range(0, 10):
    obj[i] = random.randrange(1, 100)

print(obj)

for i in range(0, 10):
  obj[i]=obj[i]**2

print(obj)

obj_slice = obj [6:10]
print(obj_slice)

for i in range(0, 10):
  if obj[i] > 500:
    print(obj[i])

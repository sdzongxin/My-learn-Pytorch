import numpy
result = []
for i in range(2,10001):
    sum = 1
    for j in range(2,i+1):
        if i%j == 0:
            sum += j
    if sum == 2*i:
        result.append(i)

print(result)
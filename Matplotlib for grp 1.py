# Matplotlib

# Visualization
import matplotlib.pyplot as plt
import numpy as np

plt.plot([1,2,3,4])
plt.ylabel('testing')
plt.show()

x = [1,2,3,4,5]
y = [10,20,30,40,50]
plt.plot(x,y)
plt.ylabel('y-axis')
plt.xlabel('x-axis')
plt.title('Random Plot')
plt.show()  # to remove text output

x = np.linspace(1,50,20)
y = np.random.randint(1,50,20)
y = np.sort(y)

plt.plot(x,y)

plt.plot(x,y,'r')
plt.plot(x,y,'red')
plt.plot(x,y,'green')
plt.plot(x,y,color = 'orange')
plt.plot(x,y,color = 'c')

####### Subplots
plt.plot()  # Convas

# 1,2,1 = 1 rows, 2 columns, 1 = graph number

plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.subplot(1,2,3) # wont work

plt.subplot(3,2,1)
plt.plot(x,y,'r')
plt.subplot(3,2,2)
plt.plot(x,y,'orange')
plt.subplot(3,2,3)
plt.plot(x,y,'black')
plt.subplot(3,2,4)
plt.plot(x,y,'b')
plt.subplot(3,2,5)
plt.plot(x,y,'c')
plt.subplot(3,2,6)
plt.plot(x,y,'g')
plt.tight_layout() # avoid overlapping

# Linestyle
plt.plot(x,y,linestyle = '-')
plt.plot(x,y,linestyle = '--')
plt.plot(x,y,linestyle = 'dotted')
plt.plot(x,y,linestyle = 'steps')

# markers
plt.plot(x,y,marker = 'o')
plt.plot(x,y,'ro')
plt.plot(x,y,color = 'r', marker = 'o')
plt.plot(x,y,color = 'r', marker = 'v')
plt.plot(x,y,color = 'r', marker = 'X')


plt.plot(x,y, color = 'r', marker = 'X', linestyle = '--')
plt.plot(x,y,'rX--')


plt.plot(x,y, color = 'r', marker = 'X', markersize = 10)

plt.plot(x,y, color = 'r', marker = 'X', markersize = 15,
         markerfacecolor = 'green', markeredgecolor = 'black',
         markeredgewidth = 3)

######
plt.plot(x,y)
plt.scatter(x,y)
plt.hist(x,y)
plt.bar(x,y)
plt.polar(x,y)

########## OOP (Object Oriented plots)
print (x)
print (y)

fig = plt.figure()
axes = fig.add_axes([0.5,0.5,1,1])
axes.plot(x,y,'r')
axes.set_xlabel('X-Axis')
axes.set_ylabel('Y-Axis')
axes.set_title('OOP')

# Graph inside Graph
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y,'g')
axes1 = fig.add_axes([0.6,0.1,0.4,0.5])
axes1.plot(y,x,'r')













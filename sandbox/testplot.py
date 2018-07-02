import matplotlib.pyplot as plt

y_val = [0.921196,0.931519,0.937498,0.934510,0.932877]
y_line=[]
for i in y_val:
	y_line.append(1-i)
x_line = [10000000,1000000,100000,10000,1000]

plt.plot(x_line,y_line)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Cross-validation error')
plt.title('SVM: Cross-validation error against C for gamma = 0.0001')
plt.show()

y_val = [0.84456522,	0.89021739,	0.90461957,	0.89157609,	0.79293478,	0.71684783,	0.64755435]
y_line=[]
for i in y_val:
	y_line.append(1-i)

x_line = [0,	0.00001,	0.0001,	0.001,	0.01,	0.1,	1]

plt.plot(x_line,y_line)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Cross-validation error')
plt.title('Perceptron: Cross-validation error against alpha')
plt.show()
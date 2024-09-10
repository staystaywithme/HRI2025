from dataprocess import reduce0, reducenoice, data301, wash_data
import numpy as np
import matplotlib.pyplot as plt

# Load data
test_a = np.loadtxt('test0.csv', delimiter=',')
test_b = np.loadtxt('test0.01.csv', delimiter=',')
test_c = np.loadtxt('test0.001.csv', delimiter=',')
test_liu = np.loadtxt('test_liuBC.csv', delimiter=',')
liuBC = np.loadtxt('liu_BC08.csv', delimiter=',')
gashiBC = np.loadtxt('gashi_BC08.csv', delimiter=',')

def view (data):
    test_processed = wash_data(data)
    test_processed = reduce0(test_processed)
    test_processed = reducenoice(test_processed)
    test_processed = data301(test_processed)
    print(test_processed.shape)

    #plot data
    plt.figure()
    for features in range (test_processed.shape[1]):
        plt.plot(test_processed[:,features],)

    plt.show()

#view(test_a)
#iew(test_b)
#view(test_c)
view(test_liu)
#view(liuBC)
#view(gashiBC)
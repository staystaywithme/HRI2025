from dataprocess import reduce0, reducenoice, data301
import numpy as np
import matplotlib.pyplot as plt

# Load data
test_a = np.loadtxt('test.csv', delimiter=',')
test_b = np.loadtxt('test0.01.csv', delimiter=',')
test_c = np.loadtxt('test0.001.csv', delimiter=',')

def view (data):
    test_processed = reduce0(data)
    test_processed = reducenoice(test_processed)
    test_processed = data301(test_processed)
    print(test_processed.shape)

    #plot data
    plt.figure()
    for features in range (test_processed.shape[1]):
        plt.plot(test_processed[:,features],)

    plt.show()

view(test_a)
view(test_b)
view(test_c)

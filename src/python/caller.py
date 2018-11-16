import sys
import itertools
import os


def DNNCaller (minCapas, maxCapas, minNeuronas, maxNeuronas):
    # genero wl arrglo con el numero de neuronas
    neuronas = []
    for i in range(minNeuronas,maxNeuronas+1):
        neuronas.append(i)
    
    for n in range(minCapas, maxCapas+1):
        for combi in itertools.product(neuronas, repeat=n):
            x = list(combi)
            x = [str(i) for i in x]
            sys.argv.append(x)
            os.system('python src/python/classifierParams.py {}'.format(" ".join(x)))

def Itercall (hidden_units, iter):
    x = [str(i) for i in hidden_units]
    for i in range(iter):
        os.system('python src/python/classifierParams.py {}'.format(" ".join(x)))

#DNNCaller(1,3,1,10)
Itercall([3], 100)
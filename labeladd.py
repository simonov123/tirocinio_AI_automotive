import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def LabelProcessing(label_b,label_c):
    '''
    algoritmo per aggiungere le lable ai dataset avversari b e c al fine di permettere l'oversampling
    Parameters
    ----------
    label_b: label del dataset B
    label_c: label del dataset C

    Returns
    -------

    '''
    print("label processing")
    #caricamento dei dataset avversari originali
    data_b=np.loadtxt('./adv_examples/dt/adv_examples_dt_bound_b.txt')
    data_c=np.loadtxt('./adv_examples/dt/adv_examples_dt_bound_c.txt')
    print(len(data_b)) # controllo
    #aggiunta della colonna delle label tramite numpy
    bound_b=np.column_stack((data_b,label_b))
    bound_c=np.column_stack((data_c,label_c))
    print("results")
    print(bound_b)
    print(bound_c)

    # Salvataggio dei dataset avversari con label pronti per ricevere oversampling
    np.savetxt(
        './adv_examples/dt/adv_examples_dt_bound_b_labeled.txt',
        bound_b,
        delimiter=' '
    )

    np.savetxt(
        './adv_examples/dt/adv_examples_dt_bound_c_labeled.txt',
        bound_c,
        delimiter=' '
    )

    return bound_b,bound_c


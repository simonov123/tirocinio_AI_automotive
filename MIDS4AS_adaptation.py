import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn_extra import cluster
import sys
from PyQt6.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout, QTextEdit, QComboBox, QLabel
import time


def boxplotshow(X):
    """
    Questo metodo restituisce un boxplot (un foglio con un boxplot per ognuna delle 13 features disposte in due colonne da 7) che mostra
    quanto ogni feature è discriminante rispetto alla label,ovvero al tipo di attacco
    Parameters
    ----------
    X: il dataset

    Returns
    -------

    """
    Features = ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11', '12', '13', '14'] #intestazione delle features
    label = 'label'             #intestazione label
    classes = X[label].unique()  #le 4 tipologie di attacco

    fig, axes = plt.subplots(7, 2, figsize=(14, 24))  # organizzazione della tabella dei plot
    axes = axes.flatten()

    for i, f in enumerate(Features):     #ciclo della creazione dei plot in ogni feature
        # valori della feature f per ciascuna classe
        data_by_class = [X[X[label] == c][f].values for c in classes]

        axes[i].boxplot(data_by_class, labels=classes)  #disposizione dei valori per le classi
        axes[i].set_title(f'Feature {f}')  #settaggio label dei plot
        axes[i].set_xlabel('Classi')
        axes[i].set_ylabel(f)

    #stampa a video dei plot
    plt.tight_layout()
    plt.show()





#Varie funzioni necessarie all'esecuzione
def kmeans_learner(X_train, k_parameter, seed):
    """
    Execute the K-Means Algorithm
    Parameters
    ----------
    X_train : X_train dataset
    k_parameter : The number of the clusters

    Returns
    -------
    kmeansobj : The prediction of the K-Means

    """
    #creazione oggetto k-means
    kmeansobj = KMeans(n_clusters=k_parameter, init='random', max_iter=300, n_init=10, random_state=seed)
    #istruzione di return
    return kmeansobj.fit(X_train)

def kMedoid_Learner(X_train2,num_clusters,metric,method,init,iter,seed):
    """

    Parameters
    ----------
    X_train: database addestramento
    num_clusters: numero di cluster predetto
    metric:metrica da usare per calcolare la distanza dal medoid
    method:algoritmo da usare, pam più lento ma più efficace
    init:inizializza il metodo con cui scegliere i medoids
    iter:max numero iterazioni
    seed:seme

    Returns
    -------

    """

    kmedoids_obj=cluster.KMedoids(num_clusters,metric,method,init,iter,seed)
    return kmedoids_obj.fit(X_train2)




def print_cluster(cluster_class, purity):
    """
       Print the clusters' information
       ----------
       cluster_class : dictionary of <cluster, class related>
       purity : the purity

       Returns
       -------
       Print the clusters' related to each class

       """
    print("\nResults:\n")
    print("Cluster: Class =", cluster_class)
    print("\n# of cluster assigned to each class:")
    print("Class 0: ", sum(1 for v in cluster_class.values() if v == 0),
          np.where(np.array(list(cluster_class.values())) == 0)[0])
    print("Class 1: ", sum(1 for v in cluster_class.values() if v == 1),
          np.where(np.array(list(cluster_class.values())) == 1)[0])
    print("Class 2: ", sum(1 for v in cluster_class.values() if v == 2),
          np.where(np.array(list(cluster_class.values())) == 2)[0])
    print("Class 3: ", sum(1 for v in cluster_class.values() if v == 3),
          np.where(np.array(list(cluster_class.values())) == 3)[0])

    print('\nPurity= ', purity)

    outpt.append("\nResults:\n")
    outpt.append(f"Cluster: Class = {cluster_class}")
    outpt.append("\n# of cluster assigned to each class:")

    outpt.append(
        f"Class 0: {sum(1 for v in cluster_class.values() if v == 0)} "
        f"{np.where(np.array(list(cluster_class.values())) == 0)[0]}"
    )

    outpt.append(
        f"Class 1: {sum(1 for v in cluster_class.values() if v == 1)} "
        f"{np.where(np.array(list(cluster_class.values())) == 1)[0]}"
    )

    outpt.append(
        f"Class 2: {sum(1 for v in cluster_class.values() if v == 2)} "
        f"{np.where(np.array(list(cluster_class.values())) == 2)[0]}"
    )

    outpt.append(
        f"Class 3: {sum(1 for v in cluster_class.values() if v == 3)} "
        f"{np.where(np.array(list(cluster_class.values())) == 3)[0]}"
    )


def class_to_cluster(y_train, kmeans_labels, class_names):
    """
    The algorithm is based on the principle of purity of the single cluster generated. For each of them,
    the algorithm retrieves the examples belonging to that cluster, then it retrieves the class associated with each
    of these examples and finally, the majority class is assigned to the cluster.

    Parameters
    ----------
    y_train : The label of the training set
    kmeans_labels : the label predicted by the K-Means
    class_names : The class used (DoS, Fuzzy, ...)

    Returns
    -------
    clusters : The clusters of the kmeans
    classToCluster : The cluster assign at each class
    pur : The purity
    """
    clusters = set(kmeans_labels)
    classes = set(y_train)
    class_to_cluster = []
    N = 0
    pur = 0

    y_train = y_train.to_list()

    for c in clusters:
        clust_examples = []
        indices = np.where(kmeans_labels == c)[0]
        for i in indices:
            clust_examples.append(y_train[i])
        maxClass = -1
        max_n_clust_example = -1

        print("\nCluster: ", c)
        for cl in classes:
            n_clust_example = clust_examples.count(cl)
            N += n_clust_example
            print(class_names[cl], ":", n_clust_example, "examples")
            outpt.append(f"{class_names[cl]} : {n_clust_example} examples")
            if n_clust_example > max_n_clust_example:
                maxClass = cl
                max_n_clust_example = n_clust_example
        pur += max_n_clust_example
        class_to_cluster.append(maxClass)
    pur = pur / N

    #DEBUG
    #print("class_to_cluster ", class_to_cluster, "type ", type(class_to_cluster))
    #print("clusters ", clusters, "type ", type(clusters))
    outpt.append("DEBUG")
    outpt.append(f"class_to_cluster {class_to_cluster} type {type(class_to_cluster)}")
    outpt.append(f"clusters {clusters} type {type(clusters)}")

    return clusters, class_to_cluster, pur

def kmeans_evaluate(X_test, kmeansobj):
    """
    Get a prediction using the K-Means fit object
    Parameters
    ----------
    X_test : The X_test dataset
    kmeansobj : The fit object of the K-Means

    Returns
    -------
    The prediction

    """
    return kmeansobj.predict(X_test)

def evaluation_results(y_test, y_pred, class_names):
    """
    Evaluate the result using the Confusion Matrix and the Classification Report
    Parameters
    ----------
    y_test : The label of the test dataset
    y_pred : The label predicted
    class_names : The class used

    Returns
    -------

    """
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion matrix:")
    outpt.append("Confusion matrix:")
    outpt.append(str(cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()
    print(classification_report(y_test, y_pred, target_names=class_names, output_dict=True))

def kmeans():
        # Esecuzione del K-means (fase di train che serve ad apprendere i centriodi
        print("[+] Exec the K-Means algorithm [Training]")
        outpt.append("[+] Exec the K-Means algorithm [Training]")
        kmeans_object = kmeans_learner(X_A, len(class_names), seed)

        # Prima del testing capiamo la: purity (number), cluster (insieme di etichette dei cluster cioè {0,1,2,3}),
        # and class_cluster (lista di 4 posizioni dove ogni posizione è la classe maggiornitaria che rappresenta il cluster)
        # questo algoritmo si basa sui risultati del k-means e sulla bontà dei cluster per portare i cluster in
        # classi da essere usate nell'apprendimento supervisionato (decision tree)
        cluster, class_cluster, purity = class_to_cluster(Y_A, kmeans_object.labels_, class_names)

        # to-do capire cos'è la purity
        # dizionario dove ogni posizione è la coppia [cluster, classe associata al cluster]
        cluster_class = dict(zip(cluster, class_cluster))

        # stampa informazioni per ogni cluster
        print_cluster(cluster_class, purity)

        # test del kmeans
        print("[+] Exec the K-Means algorithm [Testing]")
        outpt.append("[+] Exec the K-Means algorithm [Testing]")
        prediction = kmeans_evaluate(X_B, kmeans_object)

        # Elaborazione (per ogni esempio di test) della classe corrispondente al cluster predetto
        # Quindi per ogni elemento dell'array "prediction" (cioè il cluster predetto per ogni esempio di test) si va all'interno
        # del dizionario cluster_class e si vede a quale classe corrisponde (quest'ultimo riassume
        # per ogni cluster la moda delle classi degli esempi di addestramento che ricadevano in quel cluster specifico)
        y_prediction = [cluster_class.get(key) for key in prediction]

        # Valutazione dei risutalti ottenuti:
        evaluation_results(Y_B, y_prediction, class_names)

        print("Parte Avversaria")
        # Valutazione per esempi avversari : Attacco Boundary su Decision Tree
        # Test avversario : Training set è sempre X_A (con Y_A)
        # mentre il test set è X_B_bound_dt (con Y_B come verità a terra)
        print("scegliere tra dataset avversario b e c")
        outpt.append("scegliere tra dataset avversario b e c")

        scelta=inpt.currentText()
        X_B_bound_dt = np.loadtxt('./adv_examples/dt/adv_examples_dt_bound_'+scelta+'.txt')

        # L'oggetto che rappresenta il "pattern" del k-means è quello del training effettuato precedentemente
        print("[+] Exec the K-Means algorithm [Testing ADV]")
        outpt.append("[+] Exec the K-Means algorithm [Testing ADV]")
        prediction = kmeans_evaluate(X_B_bound_dt, kmeans_object)

        # Si sfrutta il mapping tra le classi (corrispondenti a ciascuna predizione e dunque a ciascun cluster predetto)
        # e i cluster precedente
        y_prediction_adv = [cluster_class.get(key) for key in prediction]
        Y_B_expanded = np.resize(Y_B, X_B_bound_dt.shape[0]) #necessario un resize del dataset avversario per evitare errori di inconsistenza dovuta alla diversa dimensione del dataset bilanciato
        #controllo lunghezza dataset
        if(len(X_A)<300000):
            evaluation_results(Y_B, y_prediction_adv, class_names)

        else:
            evaluation_results(Y_B_expanded, y_prediction_adv, class_names)


def kmedoids():
    # Esecuzione del K-means (fase di train che serve ad apprendere i centriodi
    print("[+] Exec the K-Medoids algorithm [Training]")
    kmedoids_object = kMedoid_Learner(X_A, len(class_names),"euclidean","alternate","build",300,seed)
    # Prima del testing capiamo la: purity (number), cluster (insieme di etichette dei cluster cioè {0,1,2,3}),
    # and class_cluster (lista di 4 posizioni dove ogni posizione è la classe maggiornitaria che rappresenta il cluster)
    # questo algoritmo si basa sui risultati del k-means e sulla bontà dei cluster per portare i cluster in
    # classi da essere usate nell'apprendimento supervisionato (decision tree)
    cluster, class_cluster, purity = class_to_cluster(Y_A, kmedoids_object.labels_, class_names)

    # to-do capire cos'è la purity
    # dizionario dove ogni posizione è la coppia [cluster, classe associata al cluster]
    cluster_class = dict(zip(cluster, class_cluster))

    # stampa informazioni per ogni cluster
    print_cluster(cluster_class, purity)

    # test del kmeans
    print("[+] Exec the K-Means algorithm [Testing]")
    prediction = kmeans_evaluate(X_B, kmedoids_object)

    # Elaborazione (per ogni esempio di test) della classe corrispondente al cluster predetto
    # Quindi per ogni elemento dell'array "prediction" (cioè il cluster predetto per ogni esempio di test) si va all'interno
    # del dizionario cluster_class e si vede a quale classe corrisponde (quest'ultimo riassume
    # per ogni cluster la moda delle classi degli esempi di addestramento che ricadevano in quel cluster specifico)
    y_prediction = [cluster_class.get(key) for key in prediction]

    # Valutazione dei risutalti ottenuti:
    evaluation_results(Y_B, y_prediction, class_names)

    print("Parte Avversaria")
    # Valutazione per esempi avversari : Attacco Boundary su Decision Tree
    # Test avversario : Training set è sempre X_A (con Y_A)
    # mentre il test set è X_B_bound_dt (con Y_B come verità a terra)
    X_B_bound_dt = np.loadtxt('./adv_examples/dt/adv_examples_dt_bound_b.txt')


def KNN_Learner(X_train2,Y_train2):
    """
    crea e addestra il modello KNN.
    Parameters
    ----------
    X_train2         variabili indipendenti
    Y_train2         variabile dipendente

    Returns          modello addestrato
    -------

    """
    Knn_obj = KNeighborsClassifier(          #definizione del classificatore
    n_neighbors=5,   #numero di oggetti vicini con cui confrontarsi
    weights="uniform",
    algorithm="ball_tree",
    leaf_size=30,
    p=2
)

    return Knn_obj.fit(X_train2,Y_train2)  #addestramento del modello

def Knn():
    """
    esecuzione knn
    Returns
    -------

    """
    outpt.append("[+] Exec the KNN algorithm [Training]")
    print("[+] Exec the KNN algorithm [Training]")
    knnstart=time.time()
    KnnOb=KNN_Learner(X_A,Y_A)  #creazione e addestramento del modello
    outpt.append("modello addestrato,attendere prego...")
    print("modello addestrato,attendere prego...")
    window.setWindowTitle("attendere prego...")

    y_prediction=KnnOb.predict(X_B)  #test del modello
    evaluation_results(Y_B, y_prediction, class_names) #stampa risultati
    print("[+] Exec the KNN algorithm [Testing ADV],attendere prego...")
    outpt.append("[+] Exec the KNN algorithm [Testing ADV],attendere prego...")
    scelta = inpt.currentText()
    X_B_bound_dt = np.loadtxt('./adv_examples/dt/adv_examples_dt_bound_' + scelta + '.txt')
    adv_prediction=KnnOb.predict(X_B_bound_dt)
    Y_B_expanded = np.resize(Y_B, X_B_bound_dt.shape[
        0])  # necessario un resize del dataset avversario per evitare errori di inconsistenza dovuta alla diversa dimensione del dataset bilanciato
    # controllo lunghezza dataset
    if (len(X_A) < 300000):
        evaluation_results(Y_B, adv_prediction, class_names)

    else:
        evaluation_results(Y_B_expanded, adv_prediction, class_names)

    knnend=time.time()
    outpt.append("time=")
    outpt.append(str(knnend-knnstart))
    window.setWindowTitle("Seleziona Algoritmo")


#Funzione main - punto di partenza del software
if __name__ == "__main__":
    plt.ion()
    # Nomi delle classi
    class_names = ["Normal", "Dos", "Fuzzy", "Impersonification"]
    # Impostazione del seed così da ri eseguire gli esperimenti
    np.random.seed(seed=42)
    seed = 42
    # caricamento file dataset e valorizzazione indice
    print("selezione database")
    print(""" 
    1)databse originale
    2)Database bilanciato con AI generativa
    """)
    dataset=""
    dbn=input()
    if dbn=='1':
        dataset = pd.read_csv('./dataset_final.csv', index_col=0)
    if dbn=='2':
        dataset = pd.read_csv('./dataset_balanced.csv', index_col=0)
    if dbn !='1' and dbn !='2':
        print("scelta non valida")
        dataset=""

    print('Dataset imported')
    # stampa dataset importato
    print(dataset)
    #reportistica
    #for i in range(14):
       # print(dataset[i].describe(),"\n")

    # lettura da .csv
    df_dataset = pd.DataFrame(data=dataset)
    # stampa dati dataset
    print("Number of examples: {exa} \n Number of attributes: {natt} \n Attributes: {att}".format(
        exa=len(df_dataset.values), natt=len(df_dataset.columns), att=df_dataset.columns))

    print("stampare i boxplot?")
    print("""
    1)si (verranno solo stampati i boxplot)
    2)no
    """)
    plotsc=input()
    if plotsc=='1':
        boxplotshow(df_dataset)




    # Split the dataset in independent variables and label
    # divisione dataset in variabili indipendti (dataset) e dipendenti (label)
    dataset = df_dataset.iloc[:, 0: 14]
    label = df_dataset.loc[:, 'label']

    print('Dataset splitted in independent and depedent variables')
    print("indipendenti")
    print(dataset)
    print("dipendenti")
    print(label)


    # Fase di scaling - Decommentare le istruzioni di un solo modo di scalare
    # Standard Scaling
    # sc_X = StandardScaler()
    # sc_X = sc_X.fit(dataset)
    # datas = sc_X.transform(dataset)
    # dataset = pd.DataFrame(datas)
    # print("Dataset along indipendent variables")
    # print(dataset)

    # MINMAX Scaling from 0 and 1
    # normalizazione in valori tra 0 e 1
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(dataset)
    dataset = pd.DataFrame(data)
    print("Dataset along indipendent variables")
    print(dataset)

    # Divisione del dataset in 60% and 40% (20%+20%)
    # divisione dataset in dataset di addestramento e dataset di test
    X_A, X_BC, Y_A, Y_BC = train_test_split(dataset, label, train_size=0.60, test_size=0.40, random_state=seed,
                                            stratify=label)

    X_B, X_C, Y_B, Y_C = train_test_split(X_BC, Y_BC, train_size=0.50, test_size=0.50, random_state=seed, stratify=Y_BC)

#interfaceInit
    app=QApplication(sys.argv)
    window=QWidget()
    ch_means=QPushButton("KMeans")
    ch_means.clicked.connect(kmeans)
    ch_doid=QPushButton("KMedoids")
    ch_doid.clicked.connect(kmedoids)
    ch_knn=QPushButton("KNN")
    ch_knn.clicked.connect(Knn)
    outpt=QTextEdit()
    outpt.setReadOnly(True)
    limpt=QLabel("dataset avversario")
    inpt=QComboBox()
    inpt.addItems(['b','c'])
    lay=QVBoxLayout()
    lay.addWidget(ch_means)
    lay.addWidget(ch_doid)
    lay.addWidget(ch_knn)
    lay.addWidget(outpt)
    lay.addWidget(inpt)
    window.setWindowTitle("Seleziona algoritmo")
    window.resize(400,750)
    window.setLayout(lay)
    window.show()
    sys.exit(app.exec())

#ho dovuto aggiornare numpy alla versione 2.3.4 e installare setuptools per usare KMedoids
#08/12/2025
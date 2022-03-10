import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from statistics import mode
import itertools as IT


def load_Data(file_name):
    with open(file_name, 'r') as input1:
        read_csv = pd.read_csv(input1, sep=" ", header=None)
        read_csv.drop(read_csv.columns[0], axis=1, inplace=True)
    list1 = np.array(read_csv)
    return list1


def checkEuclidean(x, y):
    a = np.sqrt(np.sum(np.subtract(x, y) ** 2))
    if a == 0:
        return True
    else:
        return False


def checkMan(x, y):
    a = np.sum(np.abs(x - y))
    if a == 0:
        return True
    else:
        return False


def pca(list1):
    List1_mean = list1 - np.mean(list1, axis=0)
    cov = np.cov(List1_mean, rowvar=False)
    eginval, eginvec = np.linalg.eigh(cov)

    sorted_index = np.argsort(eginval)[::-1]
    sort_eginval = eginval[sorted_index]
    sorted_eigenvectors = eginvec[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:len(list1)]

    n_components = 4
    eginvec_final = sorted_eigenvectors[:, 0:n_components]
    list1_reduced = np.dot(eginvec_final.transpose(), List1_mean.transpose()).transpose()

    return list1_reduced


def Euclidean(x, y):
    distance = list()
    for index, i in enumerate(y):
        distance.append(np.sqrt(np.sum((np.subtract(x, i)) ** 2)))
    return np.argmin(distance)


def manhattan(x, y):
    distance = list()
    for index, i in enumerate(y):
        distance.append(np.sum(np.abs(np.subtract(x, i))))
    return np.argmin(distance)


def K_means(k, list_, LAI, C, C2):
    itr = 0
    check = len(list_[0])
    total_len = len(list_)
    cen_index = np.random.choice(total_len, k, replace=False)
    cen = [list_[i] for i in cen_index]
    while 1:
        itr += 1
        cluster_list, cluster_classes = get_close(cen, list_, k, LAI, C)
        old_cen = cen
        cen = get_Ncen(cluster_list, k, check)

        if C2(cen, old_cen):
            # print("conv after : ", itr, " itr")
            break
    return cluster_list, cluster_classes


def get_Ncen(clusters, k, check):
    centroids = np.zeros((k, check))
    for cluster_idx, cluster in enumerate(clusters):
        cluster_mean = np.mean(cluster, axis=0)
        centroids[cluster_idx] = cluster_mean
    return centroids


def get_close(cen, list_, k, LAI, C):
    cluster_list = [[] for i in range(k)]
    cluster_classes = [[] for i in range(k)]
    for index, x in enumerate(list_):
        cluster_index = C(x, cen)
        cluster_list[cluster_index].append(x)
        cluster_classes[cluster_index].append(LAI[index][0])
    return cluster_list, cluster_classes


def get_majority(cluster_class):
    majority_list = list()
    for i in cluster_class:
        majority_list.append(mode(i))


def Get_Tp_Fp(cluster_labels):
    Tp = 0
    Fp = 0
    for i in cluster_labels:
        for a, b in IT.combinations(i, 2):
            if a == b:
                Tp += 1
            else:
                Fp += 1
    return Tp, Fp


def get_Freq(cluster_class):
    freq = np.array([0, 0, 0, 0])
    for i in range(1, 5):
        freq[i-1] = cluster_class.count(i)
    return freq


def get_Fn(freq):
    Fn = 0
    for i in range(len(freq) + 1):
        list_sum = [0, 0, 0, 0]
        for j in range(i+1, len(freq)):
            list_sum = np.add(freq[j], list_sum)
        if i != len(freq):
            Fn += np.multiply(freq[i], list_sum)
    return Fn.sum()


def get_Fn_test(freq):
    Fn = 0
    for i in range(len(freq) + 1):
        list_sum = [0, 0, 0, 0]
        for j in range(i+1, len(freq)):
            list_sum = np.add(freq[j], list_sum)
        # print("add : ", list_sum)
        if i != len(freq):
            Fn += np.multiply(freq[i], list_sum)
    return Fn.sum()


def plot_RI(R, P, F, R_norm,  P_norm, F_norm, P_man, R_man, F_man, P_Norm_man, R_Norm_man, F_Norm_man,
            P_pca_man, R_pca_man, F_pca_man, P_pca, R_pca, F_pca):
    figure, axis = pt.subplots(2, 3)
    axis[0, 0].plot(P, label="Precision")
    axis[0, 0].plot(R, label="Recall")
    axis[0, 0].plot(F, label="F-score")
    pt.xlabel("K")

    axis[0, 1].plot(P_norm, label="Precision")
    axis[0, 1].plot(R_norm, label="Recall")
    axis[0, 1].plot(F_norm, label="F-score")
    pt.xlabel("K")
    axis[1, 0].plot(P_man, label="Precision")
    axis[1, 0].plot(R_man, label="Recall")
    axis[1, 0].plot(F_man, label="F-score")
    pt.xlabel("K")

    axis[1, 1].plot(P_Norm_man, label="Precision")
    axis[1, 1].plot(R_Norm_man, label="Recall")
    axis[1, 1].plot(F_Norm_man, label="F-score")
    pt.xlabel("K")

    axis[1, 2].plot(P_pca_man, label="Precision")
    axis[1, 2].plot(R_pca_man, label="Recall")
    axis[1, 2].plot(F_pca_man, label="F-score")
    pt.xlabel("K")

    axis[0, 2].plot(P_pca, label="Precision")
    axis[0, 2].plot(R_pca, label="Recall")
    axis[0, 2].plot(F_pca, label="F-score")
    pt.xlabel("K")

    axis[0, 1].set_title("with norm Euclidean")
    axis[0, 0].set_title("without l2 norm")
    axis[1, 1].set_title("with norm Manhattan")
    axis[1, 0].set_title("without l2 norm Manhattan")
    axis[0, 2].set_title("pca Euclidean")
    axis[1, 2].set_title("pca Manhattan")

    pt.legend()
    pt.show()


def plot(list_in, list_, cen):
    fig, ax = pt.subplots(figsize=(12, 8))

    for i, index in enumerate(list_in):
        point = list_[index].T
        ax.scatter(*point, label=i + 1)

    for point in cen:
        ax.scatter(*point, marker="o", color="black", linewidth=2)
    pt.legend()
    pt.show()


def Norm(list_all):
    new_all = list()
    for index, i in enumerate(list_all):
        new_all.append(np.linalg.norm(i, 2))
    new_all = np.reshape(new_all, (len(list_all), 1 ))
    return new_all


def Recall(Tp, Fn):
    R = (Tp / (Tp + Fn))
    return R


def F_score(P, R):
    F = ((2 * P * R)/(P + R))
    return F


def Precision(Tp, Fp):
    P = (Tp / (Tp + Fp))
    return P


def Run(list_all, label_all_index, choise, choise2):
    P = list()
    R = list()
    F = list()
    for i in range(1, 11):
        freq = list()
        fin_clus, cluster_classes = K_means(i, list_all, label_all_index, choise, choise2)
        Tp, Fp = Get_Tp_Fp(cluster_labels=cluster_classes)
        P.append(Precision(Tp, Fp))
        for j in range(len(cluster_classes)):
            freq.append(get_Freq(cluster_class=cluster_classes[j]))
        Fn = get_Fn(freq)
        R.append(Recall(Tp, Fn))
        F.append(F_score(P[i-1], R[i-1]))
    return P, R, F


def main():
    # test_fn = [[5, 1, 0], [1, 4, 1], [2, 0, 3]]
    # test_Tp_Fp = [[1, 1, 1, 1, 1, 2], [1, 2, 2, 2, 2, 3], [1, 1, 3, 3, 3]]
    # print(Get_Tp_Fp(test_Tp_Fp))
    list1 = load_Data("animals")
    list2 = load_Data("fruits")
    list3 = load_Data("countries")
    list4 = load_Data("veggies")

    list_all = np.concatenate((list1, list2, list3, list4))
    list1.fill(1); list2.fill(2); list3.fill(3); list4.fill(4)

    label1_Norm = [[1] for i in range(len(list1))]; label2_Norm = [[2] for i in range(len(list2))]
    label3_Norm = [[3] for i in range(len(list3))]; label4_Norm = [[4] for i in range(len(list4))]

    label1_pca = [[1, 1] for i in range(len(list1))]; label2_pca = [[2, 2] for i in range(len(list2))]
    label3_pca = [[3, 3] for i in range(len(list3))]; label4_pca = [[4, 4] for i in range(len(list4))]

    label_all_index_pca = np.concatenate((label1_pca, label2_pca, label3_pca, label4_pca))  # 2D

    label_all_index_norm = np.concatenate((label1_Norm, label2_Norm, label3_Norm, label4_Norm))  # 1D
    label_all_index = np.concatenate((list1, list2, list3, list4))  # 300D

    P, R, F = Run(list_all, label_all_index, Euclidean, checkEuclidean)
    P_man, R_man, F_man = Run(list_all, label_all_index, manhattan, checkMan)

    list_all_Norm = Norm(list_all)
    list_all_pca = pca(list_all)
    P_pca, R_pca, F_pca = Run(list_all_pca, label_all_index_pca, Euclidean, checkEuclidean)
    P_pca_man, R_pca_man, F_pca_man = Run(list_all_pca, label_all_index_pca, manhattan, checkMan)
    P_norm, R_norm, F_norm = Run(list_all_Norm, label_all_index_norm, Euclidean, checkEuclidean)

    P_Norm_man, R_Norm_man, F_Norm_man = Run(list_all_Norm, label_all_index_norm, manhattan, checkMan)
    plot_RI(R, P, F, R_norm,  P_norm, F_norm, P_man, R_man, F_man, P_Norm_man, R_Norm_man, F_Norm_man,
            P_pca_man, R_pca_man, F_pca_man, P_pca, R_pca, F_pca)


if __name__ == '__main__':
    main()
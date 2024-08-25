import numpy as np
import ipdb
import matplotlib.pyplot as plt


def random_list(seed, length):
    seed = str(seed)
    random_list = []
    for i in range(length):
        hash_value = abs(hash(seed)) % 1e6/ 1e6
        random_list.append(hash_value)
        seed = str(seed)+ str(hash_value)
    return random_list
print(random_list(42, 15))


def create_transformed_gussain_dist(points):
    guassian_dist = np.random.randn(2,points)
    scaler = np.array([[2, 0], [0, 0.5]])
    center = np.array([2,1]).reshape(2,1)

    theta = np.pi/3
    Rotation_matrix = [[np.cos(theta), - np.sin(theta)],
                    [np.sin(theta), - np.cos(theta)]]

    tranformed = scaler @ Rotation_matrix @ guassian_dist + center

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.scatter(guassian_dist[0,:], guassian_dist[1,:])
    plt.title("Original Guassian Distribution")
    plt.xlim(-8,6)
    plt.ylim(-8,6)

    plt.subplot(1,2,2)
    plt.scatter(tranformed[0,:], tranformed[1,:])
    plt.title("Transfered Guassian Distribution")
    plt.xlim(-8,6)
    plt.ylim(-8,6)
    plt.show()
    return 
create_transformed_gussain_dist(1000)


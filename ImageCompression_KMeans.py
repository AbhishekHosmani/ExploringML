import numpy as np
import random
from matplotlib import pyplot as io
import os
import argparse
import numpy.matlib

class KMeans(object): 
    def __init__(self, image_data, file_name, k=10, n_iterations=100):
        self.data = image_data
        self.k = k
        self.n_iterations = n_iterations
        self.f_name = file_name

    def prepare_image_vectors(self):
        rows = self.data.shape[0]
        cols = self.data.shape[1]
        vec_data = self.data.reshape(rows*cols, 3)
        return vec_data
            
    def init_centroids(self, data):
        return random.sample(list(data),self.k)
    
    def get_closest_centroids(self, data, centers):
        _size = np.size(data,0)
        idx = np.zeros((_size,1))
        record = np.empty((_size,1))
        for i in range(0,self.k):
            center = centers[i]
            sq_vec = np.power(np.subtract(data,center),2)
            distance = np.sum(sq_vec,axis = 1)
            distance.resize((_size,1))
            record = np.append(record, distance, axis=1)
        record = np.delete(record,0,axis=1)
        idx = np.argmin(record, axis=1)
        return idx

    def compute_centroids(self, data, idx):
        n = np.size(data,1)
        centroids = np.zeros((self.k, n))
        for i in range(0,self.k):
            ci = idx==i
            ci = ci.astype(int)
            total_number = sum(ci);
            ci.resize((np.size(data,0),1))
            total_matrix = numpy.matlib.repmat(ci,1,n)
            ci = np.transpose(ci)
            total = np.multiply(data,total_matrix)
            centroids[i] = (1/total_number)*np.sum(total,axis=0)
        return centroids   
    
    def create_clusters(self, data, initial_centroids):
        m = np.size(data,0)
        n = np.size(data,1)
        centroids = initial_centroids
        previous_centroids = centroids
        idx = np.zeros((m,1))
        for i in range(1,self.n_iterations):
            idx = self.get_closest_centroids(data, centroids)
            centroids = self.compute_centroids(data, idx)
        return centroids,idx

    def save_image(self, image_data):
        img_name = 'output_'+ str(self.f_name) + '_k_' + str(self.k) + '_i_' + str(self.n_iterations) +'.jpg'
        io.imsave('output_'+ str(self.f_name) + '_k_' + str(self.k) + '_i_' + str(self.n_iterations) +'.jpg', image_data)
        print("Image {} Compressed with no_of_iterations = {} and k = {}".format(self.f_name, self.n_iterations, self.k))
        image_compressed = io.imread('output_'+ str(self.f_name) + '_k_' + str(self.k) + '_i_' + str(self.n_iterations) +'.jpg')
        return os.path.getsize(img_name)
    
    def compressed_size(self, image):
        return (image.size)
    
    def fit(self):
        n_rows = self.data.shape[0]
        n_cols = self.data.shape[1]
        compressed_img = np.zeros((n_rows, n_cols, 3), dtype = np.uint8)
        image_vec = self.prepare_image_vectors()
        initial_centroids = self.init_centroids(image_vec)
        centroids,idx = self.create_clusters(image_vec, initial_centroids)
        idx = self.get_closest_centroids(image_vec,centroids)
        idx = idx.reshape(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                compressed_img[i, j, :] = centroids[idx[i, j], :]
        file_size = self.save_image(compressed_img)
        return file_size
    
    
if __name__ == "__main__":
    ### Change Location Here ###
    data_loc = "/Users/abhishekhosmani/CompSci/College/ML_Anjum/data/assignment5/"
    summary = {}
    
    def visualize(compression):
        io.figure(figsize=(10,5))
        koala = io.plot(list(compression['koala.jpg'].keys()), list(compression['koala.jpg'].values()), label='koala', marker="s", linewidth=3)
        penguine = io.plot(list(compression['Penguins.jpg'].keys()), list(compression['Penguins.jpg'].values()), label='Penguine', marker="o", linewidth=3)
        io.legend()
        io.ylabel(" File Size (in Bytes)")
        io.xlabel(" No of Clusters (K) ")
        io.show()
    
    files = ['koala.jpg', 'Penguins.jpg']
    for image_file in files:
        performance = {}
        image = io.imread(data_loc+image_file)

        for _k in [2,5,10,15,20]:
            model = KMeans(image, k=_k, n_iterations=2, file_name=image_file)
            performance[_k] = model.fit()
        print("Finished Compressing the ", image_file)
        summary[image_file] = performance
    visualize(summary)
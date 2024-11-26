import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import scipy
from numpy.linalg import norm
### TODO: import any other packages you need for your solution

def feature_expansion(X, c):
    x_ = np.broadcast_to(X[:,None,:],(X.shape[0], c.shape[0], X.shape[1]))
    diffs = x_ - c
    f1 = np.linalg.norm(diffs, ord=2, axis=2)
    f2 = (X@c.T)/ \
        (norm(X, ord=2, axis=1, keepdims=True)@(norm(c, ord=2, axis=1)[None,:]))
    new_features = np.concat([f1, f2], axis=1)
    return new_features


#--- Task 1 ---#
class OneAgainstAll:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.a_s = None
        self.centroids = None
    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        n, _ = trainX.shape
        X = trainX
        Y = trainY
        
        self.centroids = []
        ##### FEATURE EXPANSION START
        for i in range(self.K):
            self.centroids.append(X[Y == i, :].mean(axis=0))
            pass
        self.centroids = np.array(self.centroids)
        new_features = feature_expansion(X, self.centroids)
        X = np.concat([X, new_features, np.ones((n,1))], axis=1)
        f = X.shape[1]
        ##### FEATURE EXPANSION END
                
        self.a_s = []        
        for i in range(self.K):
            c = np.concat([np.zeros(f), np.ones(n)])
            S = np.ones(n)
            S[Y != i] = -1
            A_ub = np.concat([-S[:,None]*X, -np.eye(n)], axis=1)
            b_ub = -np.ones(n)
            bounds = [(None, None)]*f + [(0, None)]*n
            res = scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds)
            assert res['success']
            self.a_s.append(res['x'][:f])
            pass
        self.a_s = np.array(self.a_s)        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        if len(testX.shape) == 1:
#             X = testX.resize((1, testX.size))
            X = testX[None,:]
        else:
            X = testX
            assert len(testX.shape) == 2
        n, _ = X.shape
        
        ##### FEATURE EXPANSION START
        new_features = feature_expansion(X, self.centroids)
        X = np.concat([X, new_features, np.ones((n,1))], axis=1)
        f = X.shape[1]
        assert X.shape[0] == n
        ##### FEATURE EXPANSION END
        
        # Return the predicted class labels of the input data (testX)
        res = np.argmax(self.a_s@X.T, axis=0) # Winners of duels
        
        if len(testX.shape) == 1:
            return res.squeeze()
        
        return res

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    
#--- Task 1 ---#
class OneAgainstOne:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.a_s = None
        self.centroids = None
    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        n, _ = trainX.shape
        X = trainX
        Y = trainY
        
        self.centroids = []
        ##### FEATURE EXPANSION START
        for i in range(self.K):
            self.centroids.append(X[Y == i, :].mean(axis=0))
            pass
        self.centroids = np.array(self.centroids)
        new_features = feature_expansion(X, self.centroids)
        X = np.concat([X, new_features, np.ones((n,1))], axis=1)
#         X = np.concat([X, np.ones((n,1))], axis=1)
        f = X.shape[1]
        ##### FEATURE EXPANSION END
        
        self.A = np.zeros((self.K, self.K, f))
        
#         self.centroids = []
        
        for i in range(self.K):
            for j in range(i+1, self.K):
                X_ = X[(Y == i) | (Y == j), :]
                Y_ = Y[(Y == i) | (Y == j)]
                n = Y_.size
#                 self.centroids.append(X[Y == i, :].mean(axis=0))
                c = np.concat([np.zeros(f), np.ones(n)])
                S = np.ones(n)
                S[Y_ != i] = -1
                A_ub = np.concat([-S[:,None]*X_, -np.eye(n)], axis=1)
                b_ub = -np.ones(n)
                bounds = [(None, None)]*f + [(0, None)]*n
                res = scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds)
                assert res['success']
#                 self.a_s.append(res['x'][:f])
                self.A[i,j] = res['x'][:f]
                pass
#         self.a_s = np.array(self.a_s)
#         self.centroids = np.array(self.centroids)
        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        if len(testX.shape) == 1:
#             X = testX.resize((1, testX.size))
            X = testX[None,:]
        else:
            X = testX
            assert len(testX.shape) == 2
        n, _ = X.shape
        
        ##### FEATURE EXPANSION START
        new_features = feature_expansion(X, self.centroids)
        X = np.concat([X, new_features, np.ones((n,1))], axis=1)
#         X = np.concat([X, np.ones((n,1))], axis=1)
        f = X.shape[1]
        assert X.shape[0] == n
        ##### FEATURE EXPANSION END
        
        # self.A = k x k x f
        # x = n x f
        # w = n x k x k
#         w = x@np.transpose(self.A)
        wT = self.A@X.T
        w = wT.T
        w -= np.transpose(w, (0,2,1)) # diagonal is all 0's
        w = np.sign(w) # We want to sum up the number of victories
        w = w.sum(axis=2) # n x k
        res = np.argmin(w, axis=1)
        
        if len(testX.shape) == 1:
            return res.squeeze()
        
        return res

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''


        # Update and teturn the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 (Option 1) ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return data_to_label
    




##########################################################################
#--- Task 3 (Option 2) ---#
class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        


        # Return an index list that specifies which features to keep
        return feat_to_keep
    
    
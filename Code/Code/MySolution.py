import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import scipy
from numpy.linalg import norm
import cvxpy as cp
### TODO: import any other packages you need for your solution

def feature_expansion(X, c):
    x_ = np.broadcast_to(X[:,None,:],(X.shape[0], c.shape[0], X.shape[1]))
    diffs = x_ - c
    f1 = np.linalg.norm(diffs, ord=1, axis=2)
#     f2 = (X@c.T)/ \
#         (norm(X, ord=2, axis=1, keepdims=True)@(norm(c, ord=2, axis=1)[None,:]))
    f2 = X@c.T
    new_features = np.concat([f1, f2], axis=1)
    return f1, f2

def feature_expansion_svd(X, U=None):
    if U is None:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        U = U[:min(6, U.shape[0]), :]
        pass
    x_ = np.broadcast_to(X[:,None,:],(X.shape[0], U.shape[0], X.shape[1]))
    diffs = x_ - U
    f1 = np.linalg.norm(diffs, ord=1, axis=2)
#     f2 = (X@U.T)/ \
#         (norm(X, ord=2, axis=1, keepdims=True)@(norm(U, ord=2, axis=1)[None,:]))
    f2 = X@U.T
    new_features = np.concat([f1, f2], axis=1)
    return (f1, f2), U



class Robin:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.w = None
        self.b = None
        self.lamb = 0.0001  # regularization parameter
        self.centroids = None
        self.U = None
    
    def unique_labels(self, trueY):
        self.uniqueL = np.sort(np.unique(trueY))

    def align_labels(self, trueY):
        res = np.array([np.where(self.uniqueL == y)[0][0] for y in trueY])
        return res
        
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        self.unique_labels(trainY)
        trainY = self.align_labels(trainY)
        n, d = trainX.shape
        
        self.centroids = []
       ##### FEATURE EXPANSION START
        for i in range(self.K):
            self.centroids.append(trainX[trainY == i, :].mean(axis=0))
            pass
        self.centroids = np.array(self.centroids)
        new_features = feature_expansion(trainX, self.centroids)
        trainX = np.concat([trainX, *new_features], axis=1)
        svd_feats, self.U = feature_expansion_svd(trainX)
        trainX = np.concat([trainX, *svd_feats], axis=1)
        d = trainX.shape[1]
        ##### FEATURE EXPANSION END
        
        self.w = [cp.Variable(d) for _ in range(self.K)]
        self.b = [cp.Variable() for _ in range(self.K)]
        eps = [cp.Variable(n) for _ in range(self.K)]
        delta = [cp.Variable() for _ in range(self.K)]

        constraints = []
        loss = 0
        for k in range(self.K):
            t = np.where(trainY == k, 1, -1)
            for i in range(n):
                constraints.append(t[i] * (self.w[k] @ trainX[i] + self.b[k]) >= 1 - eps[k][i])
                constraints.append(eps[k][i] >= 0)
                loss += eps[k][i]
            
            # L1 Regularization
            constraints.append(cp.abs(self.w[k]) <= delta[k])
            loss += self.lamb * delta[k]
            
        
        problem = cp.Problem(cp.Minimize(loss), constraints)
        problem.solve(verbose=False)
        self.w = [w.value for w in self.w]
        self.b = [b.value for b in self.b]
#         print("Yo")
        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        n = testX.shape[0]

#        ##### FEATURE EXPANSION START
        new_features = feature_expansion(testX, self.centroids)
        testX = np.concat([testX, *new_features], axis=1)
        svd_feats, _ = feature_expansion_svd(testX, self.U)
        testX = np.concat([testX, *svd_feats], axis=1)
        d = testX.shape[1]
        ##### FEATURE EXPANSION END
        
        scores = np.zeros((n, self.K))
        for k in range(self.K):
            scores[:,k] = testX @ self.w[k] + self.b[k]
        predY = np.argmax(scores, axis=1)

        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        testY = self.align_labels(testY)
        predY = self.predict(testX)

        accuracy = accuracy_score(testY, predY)

        return accuracy

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
        n, f_ = trainX.shape
        X = trainX
        Y = trainY
        
        self.centroids = []
        ##### FEATURE EXPANSION START
        for i in range(self.K):
            self.centroids.append(X[Y == i, :].mean(axis=0))
            pass
        self.centroids = np.array(self.centroids)
        new_features = feature_expansion(X, self.centroids)
        X = np.concat([X, *new_features], axis=1)
        svd_feats, self.U = feature_expansion_svd(X)
        X = np.concat([X, *svd_feats], axis=1)
        X = np.concat([X, np.ones((n,1))], axis=1)
        f = X.shape[1]
        ##### FEATURE EXPANSION END
                
        self.a_s = []        
        for i in range(self.K):
#             c = np.concat([np.zeros(f), np.ones(n)])
            S = np.ones(n)
            S[Y != i] = -1
#             A_ub = np.concat([-S[:,None]*X, -np.eye(n)], axis=1)
#             b_ub = -np.ones(n)
#             bounds = [(None, None)]*f + [(0, None)]*n
#             res = scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds)
#             assert res['success']
#             self.a_s.append(res['x'][:f])
            phi = cp.Variable(n)
            a = cp.Variable(f)
            constraints = [
                phi >= 0,
                (-S[:,None]*X)@a - phi <= -np.ones(n)
            ]
            objective = cp.Minimize(cp.sum(phi) + 0.05*cp.sum(cp.abs(a[:-1])))
            prob = cp.Problem(objective, constraints)
            _ = prob.solve(solver=cp.SCIPY)
            # print(x.value)
            self.a_s.append(a.value)
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
        X = np.concat([X, *new_features], axis=1)
        svd_feats, _ = feature_expansion_svd(X, self.U)
        X = np.concat([X, *svd_feats], axis=1)
        X = np.concat([X, np.ones((n,1))], axis=1)
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
        X = np.concat([X, *new_features], axis=1)
        svd_feats, self.U = feature_expansion_svd(X)
        X = np.concat([X, *svd_feats], axis=1)
        X = np.concat([X, np.ones((n,1))], axis=1)
        f = X.shape[1]
        ##### FEATURE EXPANSION END
        
        self.A = np.zeros((self.K, self.K, f))
        
#         self.centroids = []
        
        for i in range(self.K):
            for j in range(i+1, self.K):
                X_ = X[(Y == i) | (Y == j), :]
                Y_ = Y[(Y == i) | (Y == j)]
                n = Y_.size
#                 c = np.concat([np.zeros(f), np.ones(n)])
#                 S = np.ones(n)
#                 S[Y_ != i] = -1
#                 A_ub = np.concat([-S[:,None]*X_, -np.eye(n)], axis=1)
#                 b_ub = -np.ones(n)
#                 bounds = [(None, None)]*f + [(0, None)]*n
#                 res = scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds)
#                 assert res['success']
#                 self.A[i,j] = res['x'][:f]
    
                S = np.ones(n)
                S[Y_ != i] = -1
                phi = cp.Variable(n)
                a = cp.Variable(f)
                constraints = [
                    phi >= 0,
                    (-S[:,None]*X_)@a - phi <= -np.ones(n)
                ]
                objective = cp.Minimize(cp.sum(phi) + 0.05*cp.sum(cp.abs(a[:-1])))
#                 objective = cp.Minimize(cp.sum(phi))
                prob = cp.Problem(objective, constraints)
                _ = prob.solve(solver=cp.SCIPY)
                # print(x.value)
                self.A[i,j] = a.value
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
        X = np.concat([X, *new_features], axis=1)
        svd_feats, _ = feature_expansion_svd(X, self.U)
        X = np.concat([X, *svd_feats], axis=1)
        X = np.concat([X, np.ones((n,1))], axis=1)
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
def kmeans_lp_y(X, C, k):

    n, f = X.shape
    Y = cp.Variable((n, k), integer=True)

    objective = cp.Minimize(cp.sum(cp.abs(X-Y@C)))

    problem = cp.Problem(objective, [0 <= Y, Y <= 1, cp.sum(Y, axis=1)==1])
    problem.solve(solver="SCIPY")
    
    return Y.value, problem.objective.value
                            
def kmeans_lp_c(X, Y, k):

    n, f = X.shape
    C = cp.Variable((k, f))

    objective = cp.Minimize(cp.sum(cp.abs(X-Y@C)))
    
    problem = cp.Problem(objective, [])
    problem.solve(solver="SCIPY")
    
    return C.value, problem.objective.value

class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        self.C = None
        self.Y = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that describe the identified clusters
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
    
    
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

#--- Task 1 ---#
class OneAgainstAll:
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.a_s = None
        self.centroids = None
        self.with_l1_loss = True
        self.with_feature_expansion = True
        self.classes = None
        
    def align_training_labels(self, Y):
        self.classes = np.sort(np.unique(Y))
        for i,c in enumerate(self.classes):
            Y[np.isin(Y, c)] = i
            pass
        return Y
        
    def align_inference_labels(self, Y):
        assert self.classes is not None
        return self.classes[Y]
    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        trainY = self.align_training_labels(trainY.copy())
        
        n, f_ = trainX.shape
        X = trainX
        Y = trainY
        
        self.centroids = []
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
            for i in range(self.K):
                self.centroids.append(X[Y == i, :].mean(axis=0))
                pass
            self.centroids = np.array(self.centroids)
            new_features = feature_expansion(X, self.centroids)
            X = np.concat([X, *new_features], axis=1)
            svd_feats, self.U = feature_expansion_svd(X)
            X = np.concat([X, *svd_feats], axis=1)
            pass
        X = np.concat([X, np.ones((n,1))], axis=1)
        f = X.shape[1]
        ##### FEATURE EXPANSION END
                
        self.a_s = []        
        for i in range(self.K):
            S = np.ones(n)
            S[Y != i] = -1
            phi = cp.Variable(n)
            a = cp.Variable(f)
            constraints = [
                phi >= 0,
                (-S[:,None]*X)@a - phi <= -np.ones(n)
            ]
            loss = cp.sum(phi)
            if self.with_l1_loss:
                loss += 0.05*cp.sum(cp.abs(a[:-1]))
                pass
            objective = cp.Minimize(loss)
            prob = cp.Problem(objective, constraints)
            _ = prob.solve(solver=cp.SCIPY)
            self.a_s.append(a.value)
            pass
        self.a_s = np.array(self.a_s)        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        if len(testX.shape) == 1:
            X = testX[None,:]
        else:
            X = testX
            assert len(testX.shape) == 2
        n, _ = X.shape
        
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
            new_features = feature_expansion(X, self.centroids)
            X = np.concat([X, *new_features], axis=1)
            svd_feats, _ = feature_expansion_svd(X, self.U)
            X = np.concat([X, *svd_feats], axis=1)
            pass
        X = np.concat([X, np.ones((n,1))], axis=1)
        f = X.shape[1]
        assert X.shape[0] == n
        ##### FEATURE EXPANSION END
        
        # Return the predicted class labels of the input data (testX)
        res = np.argmax(self.a_s@X.T, axis=0) # Winners of duels
        res = self.align_inference_labels(res)
        if len(testX.shape) == 1:
            return res.squeeze()
        
        return res

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

class OneAgainstOne:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.a_s = None
        self.centroids = None
        self.with_l1_loss = True
        self.with_feature_expansion = True
        self.classes = None
        
    def align_training_labels(self, Y):
        self.classes = np.sort(np.unique(Y))
        for i,c in enumerate(self.classes):
            Y[np.isin(Y, c)] = i
            pass
        return Y
        
    def align_inference_labels(self, Y):
        assert self.classes is not None
        return self.classes[Y]
    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        trainY = self.align_training_labels(trainY.copy())
        n, _ = trainX.shape
        X = trainX
        Y = trainY
        
        self.centroids = []
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
            for i in range(self.K):
                self.centroids.append(X[Y == i, :].mean(axis=0))
                pass
            self.centroids = np.array(self.centroids)
            new_features = feature_expansion(X, self.centroids)
            X = np.concat([X, *new_features], axis=1)
            svd_feats, self.U = feature_expansion_svd(X)
            X = np.concat([X, *svd_feats], axis=1)
            pass
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
                S = np.ones(n)
                S[Y_ != i] = -1
                phi = cp.Variable(n)
                a = cp.Variable(f)
                constraints = [
                    phi >= 0,
                    (-S[:,None]*X_)@a - phi <= -np.ones(n)
                ]
                loss = cp.sum(phi)# + 0.05*cp.sum(cp.abs(a[:-1])))
                if self.with_l1_loss:
                    loss += 0.05*cp.sum(cp.abs(a[:-1]))
                    pass
                objective = cp.Minimize(loss)
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
        if self.with_feature_expansion:
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
        res = self.align_inference_labels(res)
        if len(testX.shape) == 1:
            return res.squeeze()
        
        return res

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    
# Task 2
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None
        self.maxIter= 100
        self.centroids = None
        self.with_feature_expansion = True
        self.U = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
            n = trainX.shape[0]
            svd_feats, self.U = feature_expansion_svd(trainX)
            trainX = np.concat([trainX, *svd_feats], axis=1)
            trainX = np.concat([trainX, np.ones((n,1))], axis=1)
            f = X.shape[1]
            pass
        ##### FEATURE EXPANSION END
        
        n, d = trainX.shape
        centroids = trainX[np.random.choice(n, self.K, replace=False)] # (K, d)
        oldCentroids = np.zeros((self.K, d))
        self.labels = np.zeros(n)

        for iter in range(self.maxIter):
            x = cp.Variable((n, self.K), integer=True) # (n, K)
            constraints = []
            _trainX = trainX[:, None, :] # (n, 1, d)
            XMinusCentroids = _trainX - centroids # (n, K, d)
            distances = np.linalg.norm(XMinusCentroids, axis=2, ord=1) # (n, K)
            
            loss = cp.sum(cp.multiply(x, distances))

            constraints.append(cp.sum(x, axis=1) == 1) # each data point belongs to one cluster
            constraints.extend([x >= 0, x <= 1]) # ensure x is binary
            problem = cp.Problem(cp.Minimize(loss), constraints)
            problem.solve(solver=cp.GLPK_MI)

            x = x.value # (n, K)
            self.labels = np.argmax(x, axis=1) # (n,)
            for k in range(self.K):
                kPoints = trainX[self.labels == k] # (n_k, d)
                assert len(kPoints) > 0  # each cluster has at least one data point
                centroids[k] = np.mean(kPoints, axis=0) # (d,)
            
            if np.allclose(centroids, oldCentroids, atol=1e-4):
                print(f'Converged at iteration {iter}')
                self.centroids = centroids
                break 

            oldCentroids = centroids.copy()

        # Update and teturn the cluster labels of the training data (trainX)
        self.centroids = centroids
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
            svd_feats, _ = feature_expansion_svd(testX, self.U)
            testX = np.concat([testX, *svd_feats], axis=1)
            testX = np.concat([testX, np.ones((n,1))], axis=1)
            f = X.shape[1]
            pass
        ##### FEATURE EXPANSION END
        
        testX_ = testX[:, None, :] # (n, 1, d)
        XMinusCentroids = testX_ - self.centroids # (n, K, d)
        distances = np.linalg.norm(XMinusCentroids, axis=2) # (n, K)
        pred_labels = np.argmin(distances, axis=1) # (n,)

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
        true_labels = true_labels.astype(int)
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
        if trainY is not None: # We can use labels for feature selection
            classes = np.unique(trainY)
            k = classes.size
            n, f = trainX.shape
            for i,c in enumerate(classes):
                trainY[np.isin(trainY, c)] = i
                pass
            l1_classifier = OneAgainstAll(k)
            l1_classifier.with_feature_expansion = False
            l1_classifier.train(trainX, trainY)
            feature_weights = np.abs(l1_classifier.a_s).sum(axis=0)
            assert len(feature_weights.shape) == 1
            feat_to_keep = feature_weights.argsort()[::-1][:self.num_features]
        else:
            k = 10
            n, f = trainX.shape
            clusterer = MyClustering(k)
            clusterer.train(X)
            Y = clusterer.infer_cluster(X)
            classifier = OneAgainstAll(k)
            classifier.with_feature_expansion = False
            classifier.train(X, Y)
            feature_weights = np.abs(l1_classifier.a_s).sum(axis=0)
            assert len(feature_weights.shape) == 1
            feat_to_keep = feature_weights.argsort()[::-1][:self.num_features]
            
        # Return an index list that specifies which features to keep
        return feat_to_keep
    
    
    
    
    
    
#--- Task 1 ---#
class OneAgainstAll_MIP:  
    def __init__(self, K, num_features):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.a_s = None
        self.centroids = None
        self.with_l1_loss = True
        self.with_feature_expansion = True
        self.num_features = num_features
    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        n, f_ = trainX.shape
        X = trainX
        Y = trainY
        
        self.centroids = []
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
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
            pass
        ##### FEATURE EXPANSION END
                
        self.a_s = []        
        for i in range(self.K):
            S = np.ones(n)
            S[Y != i] = -1
            phi = cp.Variable(n)
            a = cp.Variable(f)
            f_s = cp.Variable(f, boolean=True)
            constraints = [
                phi >= 0,
                cp.abs(a) <= f_s * 1000,
                cp.sum(f_s) == self.num_features,
                (-S[:,None]*X)@a - phi <= -np.ones(n)
            ]
            loss = cp.sum(phi)
#             if self.with_l1_loss:
# #                 loss += 0.05*cp.sum(cp.abs(a[:-1]))
#                 loss += 0.05*cp.sum(f_s)
#                 pass
            objective = cp.Minimize(loss)
            prob = cp.Problem(objective, constraints)
            _ = prob.solve(solver=cp.SCIPY)
            self.a_s.append(a.value)
            pass
        self.a_s = np.array(self.a_s)        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        if len(testX.shape) == 1:
            X = testX[None,:]
        else:
            X = testX
            assert len(testX.shape) == 2
        n, _ = X.shape
        
        ##### FEATURE EXPANSION START
        if self.with_feature_expansion:
            new_features = feature_expansion(X, self.centroids)
            X = np.concat([X, *new_features], axis=1)
            svd_feats, _ = feature_expansion_svd(X, self.U)
            X = np.concat([X, *svd_feats], axis=1)
            X = np.concat([X, np.ones((n,1))], axis=1)
            pass
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
    
    

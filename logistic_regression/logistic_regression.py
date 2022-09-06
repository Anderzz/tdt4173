import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = None
        self.C = 0.1
        
    def fit(self, X, y, lr=0.01, batch_size = 16, epochs=1500, verbose=False):
        """
        Estimates parameters for the classifier with mini-batch gradient descent
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        #setup
        m, n = X.shape
        X = np.array(X)
        y = np.array(y)
        self.w = np.zeros(n)
        self.b = 0
        self.losses = []
        #training loop
        for epoch in range(epochs):
            #mini-batch gradient descent
            for i in range((m-1)//batch_size + 1):
                #get batch
                X_batch = X[i*batch_size:(i+1)*batch_size]
                y_batch = y[i*batch_size:(i+1)*batch_size]
                #forward pass
                z = np.einsum('ij,j->i', X_batch, self.w) + self.b
                #apply activation function
                y_pred = sigmoid(z)
                #get the gradients
                dldw, dldb = self.gradients(X_batch, y_batch, y_pred)
                #do gradient descent
                self.w = self.w - lr * dldw
                self.b = self.b - lr * dldb
            #compute loss and append to list
            z = np.einsum('ij,j->i', X, self.w) + self.b
            y_pred = sigmoid(z)
            loss = binary_cross_entropy(y, y_pred)
            self.losses.append(loss)
            if verbose:
                print(f"Epoch {epoch+1} loss: {loss}")  

        return self.w, self.b
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        #forward pass
        z = np.einsum('ij,j->i', X, self.w) + self.b
        y_pred = sigmoid(z)
        return y_pred
        
    def gradients(self, X, y, y_pred, C=0.1):
        m = X.shape[0]
        #gradient of loss wrt w
        dldw = np.einsum('ij,i->j', X, y_pred - y) / m
        #gradient of loss wrt b
        dldb = np.sum((y_pred - y)) / m
        return dldw, dldb

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))



def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

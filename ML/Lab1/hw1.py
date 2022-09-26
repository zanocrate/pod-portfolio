import matplotlib.pyplot as plt
import csv

import numpy as np
import scipy as sp
import sklearn as sl
from scipy import stats
from sklearn import datasets
from sklearn import linear_model

# color list for plots
colors=[
    'darkred', 
    'navy',
    'green',
    'slateblue',
    'coral'
]


#2057447
IDnumber = 123415 # reproducibility
np.random.seed(IDnumber)

# Load the dataset
filename = 'data/music.csv'
music = csv.reader(open(filename, newline='\n'), delimiter=',')

header = next(music) # skip first line
print(f"Header: {header}\n")

dataset = np.array(list(music))
print(f"Data shape: {dataset.shape}\n")
print("Dataset Example:")
print(dataset[:10,...])

X = dataset[:,:-1].astype(float) #columns 0,1,2 contain the features
Y = dataset[:,-1].astype(int)    # last column contains the labels

print(X.shape)
print(Y.shape)

                                 # for the dataset, classical--> 0, metal --> 1
Y = 2*Y-1                        # for the perceptron classical--> -1, metal-->1
m = dataset.shape[0]
print("\nNumber of samples loaded:", m)

# ------- INITIAL PLOTTING

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X[Y>=0][:,0],X[Y>=0][:,1],X[Y>=0][:,2],color=colors[0],alpha=0.5,label='Metal') #  metal
ax.scatter(X[Y<=0][:,0],X[Y<=0][:,1],X[Y<=0][:,2],color=colors[1],alpha=0.5,label='Classical') # classical
ax.legend()


plt.show()



# Divide in training and test: make sure that your training set
# contains at least 10 elements from class 1 and at least 10 elements
# from class -1! If it does not, modify the code so to apply more random
# permutations (or the same permutation multiple times) until this happens.
# IMPORTANT: do not change the random seed.

n_classical_train=0
n_metal_train=0
while (n_classical_train < 10) and (n_metal_train < 10):

    permutation = np.random.permutation(m) # random permutation

    X = X[permutation]
    Y = Y[permutation]
    # m_test needs to be the number of samples in the test set
    m_training = int(m*0.75)

    # m_test needs to be the number of samples in the test set
    m_test = m - m_training

    # X_training = instances for training set
    X_training = X[:m_training]
    #Y_training = labels for the training set
    Y_training = Y[:m_training]

    # X_test = instances for test set
    X_test = X[m_training:]
    # Y_test = labels for the test set
    Y_test = Y[m_training:]


    print(Y_training) # to make sure that Y_training contains both 1 and -1
    print(m_test)

    n_classical_train = np.sum(Y_training==-1)
    n_metal_train = np.sum(Y_training==1)
    n_classical_test=np.sum(Y_test==-1)
    n_metal_test = np.sum(Y_test==1)

    print("\nNumber of classical instances in test:", n_classical_test)
    print("Number of metal instances in test:", n_metal_test)

    print("Shape of training set: " + str(X_training.shape))
    print("Shape of test set: " + str(X_test.shape))





# Add a 1 to each sample (homogeneous coordinates)
X_training = np.hstack((np.ones((m_training,1)),X_training))
X_test = np.hstack((np.ones((m_test,1)),X_test))

print("Training set in homogeneous coordinates:")
print(X_training[:10])


# lets try to normalize data

X_tr_norm=sl.preprocessing.normalize(X_training,axis=0)

print("Training set in normalized homogeneous coordinates:")
print(X_tr_norm[:10])


# PERCEPTRON CLASS

class Perceptron:

    def __init__(self,X,Y,Nmax,debug=False,lr=1):
        """
        Perceptron object is initialized with training.
        X = features array, shaped like (Nsamples,Nfeatures)
            needs to be expressed in homogeneous coordinates, so that
            Nfeatures has an additional 1 as first coordinate.
        Y = labels array, shaped like (Nsamples); its values are either 1 or -1
        """
        self.w = np.zeros(X.shape[1])
        self.best_w = self.w
        
        if debug:
            print("Initialized w as a zero vector")
            print(self.w)

        t = 0

        if debug: print("Calculating initial loss")
        
        loss, misclassified = self.loss(X,Y,debug)

        self.best_loss = loss
        
        if debug:
            print("Total loss = ",loss)
            print("Array of misclassifications (lenght = {}):".format(misclassified.shape))
            print(misclassified)

        while (t<abs(Nmax)) and (loss>0):
            t += 1


            random_index = np.random.randint(len(X[misclassified]))

            # ---------------------------------DEBUG
            # check progress every 500 steps
            if debug and t%500==0:
                print("Chosen point was number {} of misclassified samples, i.e.:".format(random_index) )
                print(X[misclassified][random_index])
                print(Y[misclassified][random_index])
                print("Total loss = ",loss)
                print("Array of misclassifications (lenght = {}):".format(misclassified.shape))
                print(misclassified)
            # ------------------------------------

            # ------------------ UPDATE
            self.update(X[misclassified][random_index],Y[misclassified][random_index],lr)

            loss,misclassified = self.loss(X,Y,debug)
            
            if loss<self.best_loss:
                self.best_w = self.w
                self.best_loss = loss

            #if debug: print("Current loss = ", loss)

        if debug: 
            print("Best loss was ",self.best_loss)
            print(f"We got {self.best_loss / X.shape[0] :.2%} wrong!")
            print("Best w was: ",self.best_w)
            


    def classify(self,x):
        # x must be [1,feature1,feature2,...]
        # if plane vector w is "facing" the data point
        # it gets classified as 1
        # otherwise as -1
        # returns 0 if on the boundary
        return np.sign(np.dot(self.w,x))

    def update(self,x,y,lr):
        """
        Update w using data point 
        x = [1,feature1,feature2,...]
        y = label {-1,1}
        with learning rate lr
        """
        self.w = self.w + lr*y*x

    def loss(self,X,Y,debug=False):
        """
        Calculates loss using current value of w over
        a data set X,Y.
        Returns total loss (equal to the number of misclassified samples)
        And array of boolean values corresponding to misclassified samples
        """
        
        #np.apply_along_axis takes a function, an axis and an array
        misclassified = np.apply_along_axis(self.classify,1,X) != Y

        loss = len(misclassified[misclassified == True])
            
        return loss, misclassified



perceptron=Perceptron(X_tr_norm,Y_training,10000)

print("Best w found, with loss {}:".format(perceptron.best_loss))
print(perceptron.best_w)


#---------------- plot plane

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')


#plotting against normalized training datapoints
ax1.scatter(X_tr_norm[Y_training>=0][:,1],X_tr_norm[Y_training>=0][:,2],X_tr_norm[Y_training>=0][:,3],color=colors[0],alpha=0.5,label='Metal') #  metal
ax1.scatter(X_tr_norm[Y_training<=0][:,1],X_tr_norm[Y_training<=0][:,2],X_tr_norm[Y_training<=0][:,3],color=colors[1],alpha=0.5,label='Classical') # classical

x_plot = np.linspace(np.min(X_tr_norm[:,1]),np.max(X_tr_norm[:,1]),10)
y_plot = np.linspace(np.min(X_tr_norm[:,2]),np.max(X_tr_norm[:,2]),10)

# building the points of the plane
xx, yy = np.meshgrid(x_plot,y_plot)
#xx,yy = np.meshgrid(X_tr_norm[::10,1],X_tr_norm[::10,2])

# when calculating, we need to multiply the intercept by the renormalized homogeneous value because sklearn.preprocessing.normalize acts
# on all coordinates. this could also be solved by first normalizing and then hstacking the 1s.
zz = (-perceptron.best_w[0]*X_tr_norm[0,0] - xx*perceptron.best_w[1] - yy*perceptron.best_w[2])/perceptron.best_w[3]

#ax1.set_zlimit(0,1)

ax1.plot_surface(xx,yy,zz,alpha=0.5)

ax1.legend()


plt.show()


# ------------------- TEST SET

#now use the w_found to make predictions on test dataset

# PLACE YOUR CODE to compute the number of errors

X_te_norm = sl.preprocessing.normalize(X_test,axis=0)

num_errors,test_misclassified = perceptron.loss(X_te_norm,Y_test)

true_loss_estimate = num_errors/m_test  # error rate on the test set
#NOTE: you can avoid using num_errors if you prefer, as long as true_loss_estimate is correct
print("Test Error of perceptron (100 iterations): {:.2%}".format(true_loss_estimate))


# ANSWER:
# apparently the entirety of the samples represent a trend that isn't entirely
# captured by the model trained on the training dataset only.
# we can try to visualize the difference between the training set and test set by plotting them both:

fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
#plotting normalized training datapoints
ax3.scatter(X_tr_norm[Y_training>=0][:,1],X_tr_norm[Y_training>=0][:,2],X_tr_norm[Y_training>=0][:,3],color=colors[0],alpha=0.5,label='Metal (training)')
ax3.scatter(X_tr_norm[Y_training<=0][:,1],X_tr_norm[Y_training<=0][:,2],X_tr_norm[Y_training<=0][:,3],color=colors[1],alpha=0.5,label='Classical (training)')
#plotting normalized test datapoints
ax3.scatter(X_te_norm[Y_test>=0][:,1],X_te_norm[Y_test>=0][:,2],X_te_norm[Y_test>=0][:,3],color=colors[4],alpha=0.5,label='Metal (test)') 
ax3.scatter(X_te_norm[Y_test<=0][:,1],X_te_norm[Y_test<=0][:,2],X_te_norm[Y_test<=0][:,3],color=colors[3],alpha=0.5,label='Classical (test)')
#plotting perceptron plane
x_plot = np.linspace(np.min(X_te_norm[:,1]),np.max(X_te_norm[:,1]),10)
y_plot = np.linspace(np.min(X_te_norm[:,2]),np.max(X_te_norm[:,2]),10)
xx, yy = np.meshgrid(x_plot,y_plot)
zz = (-perceptron.best_w[0]*X_tr_norm[0,0] - xx*perceptron.best_w[1] - yy*perceptron.best_w[2])/perceptron.best_w[3]

ax3.plot_surface(xx,yy,zz,alpha=0.5,color=colors[2])
ax3.legend()
ax3.view_init(azim=26., elev=15)

zmin=np.min([np.min(X_tr_norm[:,3]),np.min(X_te_norm[:,3])])
zmax=np.max([np.max(X_te_norm[:,3]),np.max(X_tr_norm[:,3])])
ax3.set_zlim(zmin,zmax)

plt.show()

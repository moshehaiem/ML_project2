import time
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BinaryClassifer():
    weight = np.random.rand(1,31)
    global train_target
    global train_data




    def __init__(self, train_data, train_target):
        """Data is already loaded for you using sklearn.datasets.load_breast_cancer
        utility function in auto-testing part. train_data is your training feature 
        data and train_target is your train label data. Hint: you can store both 
        training features and target vector as class variables"""
        
        bias = np.ones((512,1))
        self.y = np.asarray(train_target.values)
        self.x = np.asarray(train_data.values)
        self.x = np.hstack((self.x, bias))
        meanX = self.x.mean(axis=0)
        self.x-= meanX
        maximum = np.max(abs(self.x), axis=0)
        maximum += 10**-4
        self.x *= 1/maximum
        self.clf = np.zeros((57,1))


    def logistic_training(self, alpha, lam, nepoch, epsilon):
        """Training process of logistic regression will happen here. User will provide
        learning rate alpha, regularization term lam, specific number of training epoches,
        and a variable epsilon to specify pre-mature end condition,
        i.e., if error < epsilon, training stops.
        The specification of these parameters is the same as that in program #1.
        You implementation must include 3-fold validation,
        but you are allowed to hardcode the minibatch size in your program.
        Hint: You can store both weight and bias as class variables,
        so other functions can directly use them"""
        


        x = np.array_split(self.x, 3, axis=0)
        y = np.array_split(self.y, 3)
        alphaarr=np.zeros(3)
        lambdaarr = np.zeros(3)
        weightarr = []
        weightarr.append(np.zeros((1,31)))
        weightarr.append(np.zeros((1,31)))
        weightarr.append(np.zeros((1,31)))

        errorval = []
        errorval.append(float('inf'))
        errorval.append(float('inf'))
        errorval.append(float('inf'))
        errorarr = np.zeros(3)
        for n in range(3):
            tempx = 0
            tempy = 0
            valx = 0
            valy = 0
            if n == 0:
                tempx = np.vstack((x[0], x[1]))
                tempy = np.hstack((y[0], y[1]))
                valx = x[2]
                valy = y[2]
            elif n == 1:
                tempx = np.vstack((x[0], x[2]))
                tempy = np.hstack((y[0], y[2]))
                valx = x[1]
                valy = y[1]
            else:
                tempx = np.vstack((x[1], x[2]))
                tempy = np.hstack((y[1], y[2]))
                valx = x[0]
                valy = y[0]



            a = 0
            l = 0
            error = float('inf')
            step_size = 10
            curr_error=0
            curr_errorval = 0
            alpha1 = int(alpha[0])
            alpha2 = int(alpha[1])
            lam1 = int(lam[0])
            lam2 = int(lam[1])
            SGDIters = len(tempx)

            #do n fold cross validation

            for l in np.arange(lam1, lam2, (lam2-lam1) /step_size):
                for a in np.arange(alpha1, alpha2, (alpha2-alpha1)/step_size):
                    cur_iter = 1
                    w_gradient = np.zeros((1, 31))
                    curr_weight = np.random.uniform(-1, 1, (1, 31))
                    while(cur_iter <= nepoch):
                        prediction = np.zeros((1, 31))
                        for j in range(SGDIters):  
                            #here we implement batch gradient descent, and implement the correct gradient step
                            # prediction += (-((self.y[j] - (1/(1 +np.exp((-np.dot(self.x[j],np.transpose(curr_weight))))))) * self.x[j]))
                            prediction += (-((tempy[j] - sigmoid((np.dot(tempx[j],np.transpose(curr_weight))))) * tempx[j]))
                            if (((j+1) % 4) == 0):
                                w_gradient = (prediction) + (l*curr_weight)
                                curr_weight += -(a*w_gradient)
                                prediction = np.zeros((1,31))
                        cur_iter += 1
                    a /= (cur_iter)**.5


                    temp = 0
                    xn=0
                    entropyiters = len(valx)
                    for i in range(entropyiters):
                        xn = sigmoid(np.dot(valx[i],np.transpose(curr_weight)))
                        if xn <= 0:
                            xn = 10**-10
                        if xn >=1:
                            xn = .99999
                        temp += -((valy[i] * np.log(xn)) + ((1 - valy[i]) * np.log(1-xn)))
                        xn=0
                    curr_errorval = temp / 512
                    if curr_errorval < errorval[n]:
                        # weightarr[n] = curr_weight
                        errorval[n] = curr_errorval
                        # alphaarr[n] = a
                        # lambdaarr[n] = l


                        
                    #here we enter the correct curr_error
                    temp = 0
                    xn=0
                    for i in range(SGDIters):
                        xn = sigmoid(np.dot(tempx[i],np.transpose(curr_weight)))
                        if xn <= 0:
                            xn = 10**-10
                        if xn >=1:
                            xn = .99999
                        temp += -((tempy[i] * np.log(xn)) + ((1 - tempy[i]) * np.log(1-xn)))
                        xn=0
                    curr_error = temp / 512
                    if curr_error < error:
                        errorarr[n] = curr_error
                        weightarr[n] = curr_weight
                        error = curr_error
                        alphaarr[n] = a
                        lambdaarr[n] = l




        optimalalpha = 0
        optimallambda = 0
        optimalweight = np.zeros((1,31))
        errorrate = float('inf')
        errorarr[0] += errorval[0]
        errorarr[0] /= 3
        errorarr[1] += errorval[1]
        errorarr[1] /= 3
        errorarr[2] += errorval[2]
        errorarr[2] /= 3
        for num, er in enumerate(errorarr):
            if er<errorrate:
                errorrate =er
                optimalalpha = alphaarr[num]
                optimallambda = lambdaarr[num]
                optimalweight = weightarr[num]



        error = float('inf')
        cur_iter = 1
        w_gradient = np.zeros((1, 31))
        curr_weight = optimalweight
        while(cur_iter <= nepoch):
            prediction = np.zeros((1, 31))
            for j in range(512):  
            #here we implement batch gradient descent, and implement the correct gradient step
            # prediction += (-((self.y[j] - (1/(1 +np.exp((-np.dot(self.x[j],np.transpose(curr_weight))))))) * self.x[j]))
                prediction += (-((self.y[j] - sigmoid((np.dot(self.x[j],np.transpose(curr_weight))))) * self.x[j]))
                if (((j+1) % 4) == 0):
                    w_gradient = (prediction) + (optimallambda*curr_weight)
                    curr_weight += -(optimalalpha*w_gradient)
                    prediction = np.zeros((1,31))
            cur_iter += 1
                        
        #here we enter the correct curr_error
        temp = 0
        xn=0
        for i in range(512):
            xn = sigmoid(np.dot(self.x[i],np.transpose(curr_weight)))
            if xn <= 0:
                xn = 10**-10
            if xn >=1:
                xn = .99999
            temp += -((self.y[i] * np.log(xn)) + ((1 - self.y[i]) * np.log(1-xn)))
            xn=0
        curr_error = temp / 512
        if curr_error < error:
            self.weight = curr_weight
            error = curr_error
        elif curr_error < epsilon:
            return self.weight
        return self.weight


   
  


    def logistic_testing(self, testX):
        """TestX is a numpy array, which is pre-determined in this hw, referring to
        test_data in auto-testing part. (Friendly remainder: test_data is a panda dataframe)
        Use your trained weight and bias to compute the predicted y values,
        Predicted y values should be 0 or 1. return the numpy array in shape n*1"""
        xTest = testX.copy()
        meanval = xTest.mean(axis=0)
        xTest-= meanval
        maxi = np.max(abs(xTest), axis=0)
        maxi += 10**-4
        xTest *= 1/maxi
        weight = np.transpose(self.weight)[0:30]
        bias = np.transpose(self.weight)[-1]
        resultVec = np.zeros(57)
        for j in range(57):
            resultVec[j] = sigmoid((np.dot(np.transpose(weight),(xTest[j])) + bias))
        counter = 0
        for i in range(57):
            if resultVec[i] >= .5:
                resultVec[i] = 1
            else:
                resultVec[i] = 0
        resarr = resultVec.reshape((57,1))
        return resarr


    def svm_training(self, gamma, C):
        """Training process of the support vector machine will happen here. We will always use Gaussian (rbf) kernel,
        and the grid search parameters are gamma and C, which are both python lists with min and max range values.
        You are allowed to use build-in grid search for SVM. Consulting documentation for specific usage.
        As softmargin SVM can handle nonlinear boundaries and outliers much better than simple logistic regression,
        we do not perform 3-fold validation here (just one training run with 90-10 training-validation split).
        Furthmore, you are allowed to use SVM's built-in grid search method.
        This function will be a "wrapper" around sklearn.svm.SVC with all other parameters take the default values.
        Please consult sklearn.svm.SVC documents to see how to use its "fit" and "predict" functions. """

        #grid search params are gamma and C (lists with min and max)
        # built in grid search for SVM
        # one training run with 90-10 training validation split
        # use fit and predict
        C1 = C[0]
        C2 = C[1]
        svc = SVC(kernel='rbf')
        param_grid = {'C': [C1, C2], 'gamma': gamma}
        self.clf = GridSearchCV(svc, param_grid, refit=True)
        self.clf.fit(train_data, train_target)


    def svm_testing(self, testX):
        """Use your trained SVM to return the numpy array in shape n*1, predicted y values should be 0 or 1"""
        return self.clf.predict(testX).reshape(57,1)


# """ Training Process: You only need to modify nepoch, epsilon of logistic training method.
# Please donot modify anything for SVM training function. Please don't add your own svm_testing
# or logistic_testing function when you submitting this HW. This is for auto-testing """
# dataset = load_breast_cancer(as_frame=True)
# """Dataset is divided into 90% and 10%, 90% for you to perform k-fold validation and 10% for
# auto-tester to validate your performance. Please Donot change random_state, which will generate
# difffernt partitions. We want to ensure fair competition among all students"""
# train_data = dataset['data'].sample(frac=0.9, random_state=0) # random state is a seed value
# train_target = dataset['target'].sample(frac=0.9, random_state=0) # random state is a seed value
# test_data = dataset['data'].drop(train_data.index)
# test_target = dataset['target'].drop(train_target.index)

# model = BinaryClassifer(train_data, train_target)
# """Only performance of logistic regression will be used for competition. However, you must implement SVM training
# and Testing, generates reasonable results, above 90%"""
# # Compute the time to do grid search on training logistic
# logistic_start = time.time()
# model.logistic_training([10**-10, 10], [10e-10, 1e10], 300, .0005)
# logistic_end = time.time()
# model.logistic_testing(test_data)
# # Compute the time to do grid search on training SVM
# svm_start = time.time()
# model.svm_training([1e-9, 1000], [0.01, 1e10])
# model.svm_testing(test_data)
# svm_end = time.time()




""" Training Process: You only need to modify nepoch, epsilon of logistic training method.
Please donot modify anything for SVM training function. Please don't add your own svm_testing
or logistic_testing function when you submitting this HW. This is for auto-testing """
dataset = load_breast_cancer(as_frame=True)
"""Dataset is divided into 90% and 10%, 90% for you to perform k-fold validation and 10% for
auto-tester to validate your performance. Please Donot change random_state, which will generate
difffernt partitions. We want to ensure fair competition among all students"""
train_data = dataset['data'].sample(frac=0.9, random_state=0) # random state is a seed value
train_target = dataset['target'].sample(frac=0.9, random_state=0) # random state is a seed value
test_data = dataset['data'].drop(train_data.index)
test_target = dataset['target'].drop(train_target.index)

model = BinaryClassifer(train_data, train_target)
"""Only performance of logistic regression will be used for competition. However, you must implement SVM training
and Testing, generates reasonable results, above 90%"""
# Compute the time to do grid search on training logistic
logistic_start = time.time()
model.logistic_training([10**-10, 10], [10e-10, 1e10], 300, .0005)
logistic_end = time.time()
# Compute the time to do grid search on training SVM
svm_start = time.time()
model.svm_training([1e-9, 1000], [0.01, 1e10])
svm_end = time.time()


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.mixture import GaussianMixture,GMM
import Feature_extraction as fe
from sklearn.metrics import confusion_matrix



def draw_ellipse(position, covariance, alpha,clr, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    #print color
    ell=mpl.patches.Ellipse(position, 2*width, 2*height,angle,color=clr, **kwargs)
    # Draw the Ellipse
    ax.add_patch(ell)
    ell.set_alpha(alpha)
    

def GMM_func(X_train, Y_train, X_test, Y_test, n_classes, show_results=False, fplt=False,colors='rgbym',select_classifier=2):
    temp1=X_train[:,0]
    temp2=X_train[:,1]
    temp1=np.reshape(temp1,(X_train.shape[0],1))
    temp2=np.reshape(temp2,(X_train.shape[0],1))
    mean1=np.array([temp1[Y_train == i].mean()
                                  for i in range(n_classes)])
    mean2=np.array([temp2[Y_train == i].mean()
                                  for i in range(n_classes)])
    #print mean1
    #print mean2

    mean_vector=np.zeros((n_classes,2))
    mean_vector[:,0]=mean1
    mean_vector[:,1]=mean2

    #print mean_vector

    # Try GMMs using different types of classifiers.

    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.

    if select_classifier==1:
        classifier1 = GMM(n_components=n_classes,covariance_type='full',init_params='wc', n_iter=200)
        classifier1.means_=mean_vector
        classifier1.fit(X_train)
        if fplt:
            w_factor = 0.5 / classifier1.weights_.max()
            for pos, covar, w, color in zip(classifier1.means_, classifier1.covars_, classifier1.weights_,colors):
                draw_ellipse(pos, covar,alpha=w*w_factor, clr=color)
        Y_pred=classifier1.predict(X_test)

    if select_classifier==2:
        classifier2 = GaussianMixture(n_components=n_classes,means_init=mean_vector,covariance_type='full', max_iter=5000)
        classifier2.fit(X_train)
        if fplt:
            w_factor = 0.8 / classifier2.weights_.max()
            for pos, covar, w,color in zip(classifier2.means_, classifier2.covariances_,classifier2.weights_,colors):
                draw_ellipse(pos,covar,alpha=w*w_factor,clr=color)
        Y_pred=classifier2.predict(X_test)
    
    Y_pred=np.reshape(Y_pred,(Y_test.shape[0],1))
    Y_pred[Y_pred!=0]=1
    Y_test[Y_test!=0]=1
    #print Y_pred
    #print Y_test
    confusion = confusion_matrix(Y_test, Y_pred)

    eps = 1e-9
    # print confusion
    accuracy = 0
    if float(np.sum(confusion))!= 0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1]+eps)
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0]+eps)
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1]+eps)
    if show_results:
        print("\nGlobal Accuracy: " +str(accuracy))
        print("Specificity: " +str(specificity))
        print("Sensitivity: " +str(sensitivity))
        print("Precision: " +str(precision))
    return accuracy

def split(arr,k,n):
    arr1, arr2 = np.zeros(len(arr)/n), np.zeros((n-1)*int(len(arr)/n))
    for i in range(0,len(arr)/5):
        arr1[i] = arr[5*i+k]
        arr2[4*i] = arr[5*i+1+k]
        arr2[4*i+1]=arr[5*i+2+k]
        arr2[4*i+2]=arr[5*i+3+k]
        arr2[4*i+3]=arr[5*i+4+k]
    return arr1, arr2
       
def split2(arr,k,n):
    arr1, arr2 = np.zeros((len(arr)/n,2)), np.zeros(((n-1)*int(len(arr)/n),2))
    for i in range(0,len(arr)/5):
        arr1[i] = arr[5*i+k]
        arr2[4*i] = arr[5*i+1+k]
        arr2[4*i+1]=arr[5*i+2+k]
        arr2[4*i+2]=arr[5*i+3+k]
        arr2[4*i+3]=arr[5*i+4+k]
    return arr1, arr2 

def cross_validation(X_train, Y_train, n_classes):
    m,show_results,index,n = 0.0,0,0,5
    acc = np.zeros(n)
    length_data = int(len(Y_train)/5)*5
    X_train = X_train[0:length_data-1]
    Y_train = Y_train[0:length_data-1]
    for k in range(0,n):
        X1, X2 = split2(X_train,k,5)
        Y1, Y2 = split(Y_train,k,5)
        acc[k] = GMM_func(X2, Y2, X1, Y1, n_classes, show_results)
        if m<acc[k]: 
            m = acc[k]
            index = k
        
    print("\nCross Validation Maximum Accuracy: " +str(m))

def data_division1(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont):
    X_train = X_neutral
    Y_train = Y_neutral
    X_test = np.append(X_Happy,X_Sad,axis=0)
    Y_test = np.append(Y_Happy,Y_Sad)
    X_test = np.append(X_test,X_cont,axis=0)
    Y_test = np.append(Y_test,Y_cont)
    return X_train,X_test,Y_train,Y_test

def data_division2(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont):
    X_train = np.append(X_Happy,X_Sad,axis=0)
    Y_train = np.append(Y_Happy,Y_Sad)
    X_train = np.append(X_train,X_neutral,axis=0)
    Y_train = np.append(Y_train,Y_neutral)
    X_test = X_cont
    Y_test = Y_cont
    return X_train,X_test,Y_train,Y_test

        
if __name__ == '__main__':
    n_classes = 4
    p = 1
    Colors = 'rgbym'
    plt.figure()
    
    #X_neutral is Neutral Data 
    print "\nGetting Neutral Data....\n"
    X_neutral,Y_neutral = fe.get_data(n_classes,neutral=True)
    print "--------------------------------------------------------"
    print X_neutral.shape
    
    #X_Happy is Emotional Data = 'Happy'
    print "\nGetting Emotional Data(Happy)....\n"
    X_Happy,Y_Happy = fe.get_data(n_classes,'Happy',cont=False)
    print "--------------------------------------------------------"
    print X_Happy.shape    

    #X_Sad is Emotional Data = 'Sad'
    print "\nGetting Emotional Data(Sad)....\n"
    X_Sad,Y_Sad = fe.get_data(n_classes,'Sad',cont=False)
    print "--------------------------------------------------------"
    print X_Sad.shape

    #X_cont is Continuous Data
    print "\nGetting Continuous Data....\n"
    X_cont,Y_cont = fe.get_data(n_classes,cont=True)
    print "--------------------------------------------------------"
    print X_cont.shape

    #Preparing the Training data
    #Training on Neutral Data and Testing on Emotional and Continuous Data

    #X_train,X_test,Y_train,Y_test=data_division1(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont)

    #Training on Neutral Data and Testing on Emotional and Continuous Data

    X_train,X_test,Y_train,Y_test=data_division2(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont)

    #For Plotting the training data points
    for n, color in enumerate(Colors):
        datatemp1=X_train[:,0]
        datatemp2=X_train[:,1]
        datatemp1=np.reshape(datatemp1,(X_train.shape[0],1))
        datatemp2=np.reshape(datatemp2,(X_train.shape[0],1))
        data1 = datatemp1[Y_train == n]
        data2 = datatemp2[Y_train == n]
        plt.scatter(data1, data2, 0.8, color=color,label=n)
    
    #Cross Validation 
    cross_validation(X_train, Y_train, n_classes)
    
    #Training on Neutral and Emotional Data and Testing on Continuous Data
    GMM_func(X_train, Y_train, X_test, Y_test, n_classes, True, True, Colors, select_classifier=2)
   

    plt.xticks(())
    plt.yticks(())

    plt.legend(loc='lower right')
    plt.show()   

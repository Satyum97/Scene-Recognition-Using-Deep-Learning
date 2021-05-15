# Importing Libraries from CNN_layers
from architecture.CNN_layers import *

# Importing ML Libraries
import sys
import numpy as np
import pickle


''' Constructing Convolution_Neural_Network Class to build the layers Framework for Convolution Neural Net '''
class Convolution_Neural_Network:
    def __init__(self):
        
        
        alpha = 0.1
        self.CNN_layers = []


        '''
        Neural Network layer            : 0
        Input to the Convolution Layer  : 100 x 100 x 3
        kernel size                     : 5 x 5 x 6
        stride                          : 1
        padding                         : 2
        Output of the Convolution Layer : 100 x 100 x 6
        '''
        self.CNN_layers.append(Convolution_Layer(inputs_channel=3, num_filters=6, kernel_size=5, padding=2, stride=1, learning_rate=alpha, name='cv1'))
        

        '''
        Neural Network layer            : 1
        Input to the  Layer             : 100 x 100 x 6
        Output of the Layer             : 100 x 100 x 6
        '''
        self.CNN_layers.append(LeakyReLuActivation())

        '''
        Neural Network layer            : 2
        Input to the Maxpooling Layer   : 100 x 100 x 6
        kernel size                     : 2 x 2
        stride                          : 1
        pool size                       : 2
        Output of the Maxpooling Layer  : 50 x 50 x 6
        '''
        self.CNN_layers.append(Maxpool_Layer(pool_size=2, stride=2, name='m1'))
        

        '''
        Neural Network layer            : 3
        Input to the Convolution Layer  : 50 x 50 x 6
        kernel size                     : 5 x 5 x 16
        stride                          : 1
        padding                         : 2
        Output of the Convolution Layer : 50 x 50 x 16
        '''
        self.CNN_layers.append(Convolution_Layer(inputs_channel=6, num_filters=16, kernel_size=5, padding=2, stride=1, learning_rate=alpha, name='cv2'))
        
        
        '''
        Neural Network layer            : 4
        Input to the Layer              : 50 x 50 x 16
        Output of the Layer             : 50 x 50 x 16
        '''
        self.CNN_layers.append(LeakyReLuActivation())


        '''
        Neural Network layer            : 5
        Input to the Flattening layer   : 50 x 50 x 16
        Output of the Flattening layer  : 1 x 40000
        '''
        self.CNN_layers.append(Flattening_Layer())

        '''
        Neural network layer            : 6
        Input to Fully Connected layer  : 1 x 40000
        Output of Fully Connected layer : 1 x 36
        '''
        self.CNN_layers.append(FullyConnected_Layer(num_inputs=40000, num_outputs=36, learning_rate=alpha, name='c1'))
        

        '''
        Neural Network layer            : 7
        Input to the Layer              : 1 x 36
        Output of the Layer             : 1 x 36
        '''        
        self.CNN_layers.append(LeakyReLuActivation())


        '''
        Neural network layer            : 8
        Input to Fully Connected layer  : 1 x 36
        Output of Fully Connected layer : 1 x 6
        '''
        self.CNN_layers.append(FullyConnected_Layer(num_inputs=36, num_outputs=6, learning_rate=alpha, name='c2'))

        
        '''
        Neural network layer            : 9
        Input to Softmax layer          : 1 x 6
        Output of Softmax layer         : 1 x 6
        '''
        self.CNN_layers.append(Softmax_Layer())
        self.numOflayers = len(self.CNN_layers)






    '''
    # Computes the Cross Entropy Loss for the Neural Network
    # params    : @inputs       : Image dataset array
                  @categories   : Class Categories array
    # returns   : Cross Entropy Loss
    '''    
    def error(self,inputs, categories):
        numOfOuptput = categories.shape[0]
        p = np.sum(categories.reshape(1,numOfOuptput)*inputs)
        e = -np.log(p)
        return e
    





    '''
    #Generates the sample input for training dataset
    # params    : @sampleInd            : sample index
                  @sample_group         : sample size
                  @train_dataset        : training dataset images
                  @generated_trainLabels: training dataset labels

    # returns   : sample input for training data and its corresponding training labels
    '''

    def sample_group_input(self,sampleInd,sample_group,train_dataset,generated_trainLabels):
        t = train_dataset.shape[0]
        s = sampleInd+sample_group
        if t > s :
            td = train_dataset[sampleInd:s]
            tl = generated_trainLabels[sampleInd:s]
            return (td,tl)
        else:
            td = train_dataset[sampleInd:t]
            tl = generated_trainLabels[sampleInd:t]
            return (td,tl) 






    '''
    # Performing weight updates from input layer to output layer
    #params     : @dataset_images : Training dataset images array
                  @dataset_labels : Training labels array
                  @Acc            : Training dataset Accuracy
                  @totalAcc       : Training dataset Total Accuracy
                  @modelError     : Training dataset Model Loss

    #returns    : ans,dataset_images,modelError,Acc,totalAcc,result
    '''
    def forward_pass(self,dataset_images,dataset_labels,Acc,totalAcc,modelError):
        for val in range(self.numOflayers):
            ans = self.CNN_layers[val].forward_propagation(dataset_images)
            dataset_images = ans
        modelError += self.error(ans, dataset_labels)
        if np.argmax(ans) == np.argmax(dataset_labels):
            Acc += 1
            totalAcc += 1
        result=ans
        return (ans,dataset_images,modelError,Acc,totalAcc,result)
    






    '''
    # Performing weight updates from output layer to input layer
    # params    : @b: class categories array

    # returns   : updated class categories array
    '''
    def backward_pass(self,b):
        for val in range(self.numOflayers-1, -1, -1):
            ans = self.CNN_layers[val].backward_propagation(b)
            b = ans
        return b      







    '''
    # Computes the Training Accuracy and the Model Loss for the Training Dataset
    # params    : @e            : current epoch
                  @totalAcc     : Training Dataset Total Accuracy
                  @train_dataset: Training Dataset
                  @epoch        : Total Epoch
                  @sampleInd    : sample index
                  @sample_group : sample size
                  @modelError   : Training Dataset Model Loss
                  @Acc          : Training Dataset Accuracy
    #returns    : modelError,groupAcc,trainingAcc
    '''
    def train_accuracy(self,e,totalAcc,train_dataset,epoch,sampleInd,sample_group,modelError,Acc):
        modelError /= sample_group
        groupAcc = float(Acc)/float(sample_group)
        trainingAcc = float(totalAcc)/float((sampleInd+sample_group)*(e+1))
        return(modelError,groupAcc,trainingAcc)







    '''
    # Training the Convolution Neural Network
    #params     : @train_dataset        : Training Dataset Array
                  @generated_trainLabels: Training Labels Array
                  @sample_group         : sample size
                  @epochs               : Number of Forward and Backward Iteration
                  @weightsFile          : Intialized Weights of CNN
    #returns    : prints the Epochs, Current Iteration, Error, Sample Group Accuracy, Training Accuracy
    '''
    def train(self, train_dataset, generated_trainLabels, sample_group, epoch, weightsFile):
        totalAcc = 0
        tds = train_dataset.shape[0]
        for e in range(epoch):
            for sampleInd in range(0, tds, sample_group):
                td,tl = self.sample_group_input(sampleInd,sample_group,train_dataset,generated_trainLabels)
                modelError = 0
                Acc = 0
                for i in range(sample_group):
                    ti = td[i]
                    tc = tl[i]
                    result,ti,modelError,Acc,totalAcc,b = self.forward_pass(ti,tc,Acc,totalAcc,modelError)
                    b = self.backward_pass(b)                   
                modelError,sample_groupAcc,trainAcc = self.train_accuracy(e,totalAcc,train_dataset,epoch,sampleInd,sample_group,modelError,Acc)
                print('-------Epoch: {0:d}/{1:d}-----Iteration:{2:d}-----Error: {3:.2f} -----Batch Accuracy: {4:.2f} -----Training Accuracy: {5:.2f} ------'.format(e,epoch,sampleInd+sample_group,modelError,sample_groupAcc,trainAcc)) 
            w = []
            for i in range(self.numOflayers):
                wt = self.CNN_layers[i].extract_weights()
                w.append(wt)
            with open(weightsFile, 'ab') as pick_file:
                pickle.dump(w, pick_file, protocol=pickle.HIGHEST_PROTOCOL)
                


    '''
    Testing the Convolution Neural Network
    #params :      @test_dataset       :Testing Dataset Array
                   @generated_testLabel:Testing Labels Array
                   @dim                :Testing Dataset Size
    #returns:   prints Testing Size and Testing Accuracy

    '''
    def test(self, test_dataset, generated_testLabel, dim):
        totalAcc = 0
        Acc = 0
        modelError = 0
        for i in range(dim):
            a = test_dataset[i]
            b = generated_testLabel[i]
            output,a,modelError,Acc,totalAcc,result = self.forward_pass(a,b,Acc,totalAcc,modelError)
        sys.stdout.write("\n")
        print('-----Test Size:{0:d}----Testing accuracy:{1:.2f}----'.format(dim, float(totalAcc)/float(test_size)))
        


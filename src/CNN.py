import numpy as np
from ConvolutionLayer import ConvLayer
from FullyConnectedLayer import FullyConnected
from  MaxPoolingLayer import MaxPool


class ConvNet:
    def __init__(self,lr,epoch):
        self.lr=lr
        self.epoch=epoch
        #Layer Definitions
        self.conv1=ConvLayer((3,3,3),5,1,0,0.01)
        self.pool1=MaxPool((2,2),1,0.01)
        self.conv2=ConvLayer((3,3,3),10,1,0,0.01)
        self.pool2=MaxPool((2,2),1,0.01)
        self.dense=FullyConnected(0.01,16,10)

    def train(self,x,y): #training method
        self.x=x
        self.y=y
        
        for _ in range(self.epoch):
            # The Forward Pass
            out1=self.pool1.forward(self.conv1.forward(self.x))
            out1=self.pool2.forward(self.conv2.forward(out1))
            out1=self.dense.forward(out1)
        
            #The backward pass

            error=self.dense.backward(self.x,self.y)
            error=self.conv2.backward(self.pool2.backward(error))
            error=self.conv1.backward(self.pool1.backward(error))
    

    def predict(self,x): #Perform New Predictions
            out1=self.pool1.forward(self.conv1.forward(self.x))
            out1=self.pool2.forward(self.conv2.forward(out1))
            return self.dense.forward(out1)


        
    def LeakyReLU(self,x): #Leaky Relu Activation
        return np.maximum(0.001,x)
    
    def softmax(self,x): #Softmax activation
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=1,keepdims=True)


    def he_initialization(self,shape): #Kaiming Initialization
        fan_in = shape[0]  
        stddev = np.sqrt(2 / fan_in)
        return np.random.randn(*shape) *stddev
    
 

            
            

    

            

from CNN import ConvNet
import numpy as np

class FullyConnected(ConvNet):
    def __init__(self,lr,neuronCount,noOfOutputs):
        super().__init__()
        self.lr=lr
        self.neuronCount=neuronCount
        self.noOfOutputs=noOfOutputs
        self.reluGrad=lambda x:np.where(x>0,x,0.01)
        self.init_weights(800) # Static  - Change this according to the number of features


    def forward(self,x):
        self.x=x
        
        #Computation for fc layer 1
        self.a1=np.dot(self.x,self.l1_w)+self.b1
        self.z1=super().LeakyReLU(self.a1)

        #Computation for fc layer 2
        self.a2=np.dot(self.z1,self.l2_w)+self.b2
        self.z2=super().LeakyReLU(self.a2)

        #Computation for fc layer 3
        self.a3=np.dot(self.z2,self.l3_w)+self.b3
        self.z3=super().softmax(self.a3)
        
        return self.z3
    
    def init_weights(self,featureCount):
        self.l1_w=super().he_initialization((featureCount,self.neuronCount))
        self.b1=np.zeros(self.neuronCount)
        self.l2_w=super().he_initialization((self.neuronCount,self.neuronCount))
        self.b2=np.zeros(self.neuronCount)
        self.l3_w=super().he_initialization((self.neuronCount,self.noOfOutputs))
        self.b3=np.zeros(self.noOfOutputs)
    
    def backward(self,x,y):
        #Gradient computation for output layer
        error3=self.a3-y
        dw3=np.dot(error3,self.a2)
        db3=np.sum(error3,axis=0,keepdims=True)

        #Take a step for gradient descent
        self.l3_w-=(self.lr*dw3)
        self.l3_b-=(self.lr*db3)

        #Gradient computation for layer2
        error2=np.dot(error1,self.l3_w.T)*self.reluGrad(self.z2)
        dw2=np.dot(error2,self.a1)
        db2=np.sum(error2,axis=0,keepdims=True)

        #Take a step for gradient descent
        self.l2_w-=(self.lr*dw2)
        self.l2_b-=(self.lr*db2)

        #Gradient computation for layer1
        error1=np.dot(error2,self.l2_w.T)*self.reluGrad(self.z1)
        dw1=np.dot(error1,x)
        db1=np.sum(error1,axis=0,keepdims=True)

        #Take a step for gradient descent
        self.l1_w-=(self.lr*dw1)
        self.l1_w-=(self.lr*db1)


        return error1
        

        

        

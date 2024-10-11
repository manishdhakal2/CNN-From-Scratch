from CNN import ConvNet
import numpy as np

class MaxPool(ConvNet):
    def __init__(self,poolSize:np.array,stride:int,lr):
        super().__init__()
        self.poolSize=poolSize
        self.stride=stride
        self.lr=lr
        self.indices=None
    
    def Pool3D(self,x):
        self.x=x
        #Calculation of Output Dimension
        pool_l=((x.shape[0]-self.poolSize[0])//self.stride)+1
        pool_w=((x.shape[1]-self.poolSize[1])//self.stride)+1 

        #Create the output array
        poolOutput=np.zeros((pool_l,pool_w,x.shape[2]))
        self.indices=np.zeros_like(x)


        for i in range(x.shape[2]): #  Loop over each channel
            for j in range(0,x.shape[0]-self.poolSize[0]+1,self.stride): # Loop over the pool area
                for k in range(0,x.shape[1]-self.poolSize+1,self.stride):
                    val=x[j:j+self.poolSize[0],k:k+self.poolSize[1],i] # extract the area to find the max value
                    
                    poolOutput[j//self.stride,k//self.stride,i]=np.max(val) #Assign the max value to the consequent index of the output

                    max_idx = np.unravel_index(np.argmax(val), val.shape)
                    self.indices[j + max_idx[0], k + max_idx[1], i] = 1 
        return poolOutput




    
    def forward(self,x):
        return self.Pool3D(x)
    

    def backward(self,error):
        dw=np.zeros_like(self.x)

        for i in range(self.x.shape[2]): #  Loop over each channel
            for j in range(0,self.x.shape[0]-self.poolSize[0]+1,self.stride): # Loop over the pool area
                for k in range(0,self.x.shape[1]-self.poolSize+1,self.stride):
                    val=self.x[j:j+self.poolSize[0],k:k+self.poolSize[1],i] # extract the area to find the max value

                    max_idx = np.unravel_index(np.argmax(val), val.shape)

                    dw[j+max_idx[0],k+max_idx[1],i]+=error[j//self.stride,k//self.stride,i]
        
        return dw
 


        

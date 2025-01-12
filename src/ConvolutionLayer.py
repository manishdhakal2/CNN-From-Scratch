from CNN import ConvNet
import numpy as np

class ConvLayer(ConvNet):
    def __init__(self,convSize,filterCount:int,stride:int,padding:int,lr):
    
        super().__init__()
        self.convSize=convSize
        self.filterCount=filterCount
        self.stride=stride
        self.padding=padding
        self.lr=lr
    
    def __initialize__(self):
        self.filters=[]
        #Create the array to store the filters
        for i in range(self.filterCount):
            arr=super().he_initialization(self.convSize)
            self.filters.append(arr)
    
    def Conv2D(self, arr, filter, padding=0, stride=1):
        # Apply Padding if necessary
        if padding > 0:
            arr = np.pad(arr, pad_width=((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
        # Calculation of dimensions
        H, W, C = arr.shape  # Height, Width, Channels
        kH, kW, _ = filter.shape  # Kernel Height, Kernel Width, Channels (same as input channels)
        output_H = (H - kH) // stride + 1
        output_W = (W - kW) // stride + 1
        output_volume = np.zeros((output_H, output_W))
    
        # Actual Convolution Process
        for y in range(0, output_H):
            for x in range(0, output_W):
                input_region = arr[y*stride:y*stride+kH, x*stride:x*stride+kW, :]  # Extract 2D region
                output_volume[y, x] = np.sum(input_region * filter)  # Element-wise multiplication and sum
    
        return output_volume

        

    
    def forward(self,x):
        self.x=x
        #Calculate the output size
        conv_L=((x.shape[0]-self.convSize[0]+2*self.padding)//self.stride)+1
        conv_W=((x.shape[1]-self.convSize[1]+2*self.padding)//self.stride)+1
        convOutput=np.zeros((conv_L,conv_W,self.filterCount))
    
        #Apply the convolution to each image

        for i in range(self.filterCount):
            convOutput[:,:,i]=self.Conv2D(x,self.filters[i],padding=0,stride=1)
        
        return convOutput
    
    def backward(self,prevError):
        prevError=prevError.reshape(self.x.shape)
        conv_L=((prevError.shape[0]-self.convSize[0]+2*self.padding)//self.stride)+1
        conv_W=((prevError.shape[1]-self.convSize[1]+2*self.padding)//self.stride)+1
        convOutput=np.zeros((conv_L,conv_W,self.filterCount))

        for i in range(self.filterCount):
            convOutput[:,:,i]=self.Conv2D(prevError,self.filters[i],padding=0,stride=1)
        
        for i in range(0,self.filterCount):
            self.filters[i]-=self.lr*convOutput[i]
        
        return convOutput
        


        



        

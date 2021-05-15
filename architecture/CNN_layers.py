import numpy as np
#This is the class for the Convolutional Neural layer. All the variables related to Convolutional
#layer are initialized in this class. When called from the networks.py file, 
#it will be used to generate an instance of the convolutional layer
class Convolution_Layer:
#params:@ input_stream      : Number of input channels
#       @ NoOfFilters       : Number of filters/kernels for convolving the image 
#       @ SizeofKernel      : Size of the kernel/filters, which will be used for convolving the image
#       @ padding           : number of padding pixels
#       @ stride            : Number of shifts to be made while convolving
#       @ lrate             : Learning rate for the model 
#       @ name              : Name of the layer
    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):

        self.Filters = num_filters
        self.Kernels = kernel_size
        self.Channels = inputs_channel

        self.weights = np.zeros((self.Filters, self.Channels, self.Kernels, self.Kernels))
        self.bias = np.zeros((self.Filters, 1))
        for i in range(0,self.Filters):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.Channels*self.Kernels*self.Kernels)), size=(self.Channels, self.Kernels, self.Kernels))

        self.pad = padding
        self.ST = stride
        self.LRate = learning_rate
        self.Layer_name = name
        
# This is the padding function for padding extra zeros to the images 
    #if the images are not able to fit the kernel convolutions 
    #params: @inputs    : image that has to be padded
    #        @size      : number of zeroes to be padded
    #return: @out       : padded image
    def zero_pad(self, inputs, size):
        wid, hei = inputs.shape[0], inputs.shape[1]
        new_wid = 2 * size + wid
        new_hei = 2 * size + hei
        out = np.zeros((new_wid, new_hei))
        out[size:wid+size, size:hei+size] = inputs
        return out

#This is the function for forward propagation in the Concolutional Layer 
    #params: @inputs            : input for forward propagation
    #return: @feature_maps      : matrix of all the features of the images
    def forward_propagation(self, inputs):
        Channels = inputs.shape[2]
        Width = inputs.shape[0]+2*self.pad
        Height = inputs.shape[1]+2*self.pad
        self.inputs = np.zeros(( Width, Height, Channels))
        for c in range(inputs.shape[2]):
            self.inputs[:,:, c] = self.zero_pad(inputs[:,:, c], self.pad)
        WW = int((Width - self.Kernels)/self.ST) + 1
        HH = int((Height - self.Kernels)/self.ST) + 1
        feature_maps = np.zeros(( WW, HH, self.Filters))

        try:
            for f in range(self.Filters):
                for ch in range(Channels):
                    for w in range(WW):
                        for h in range(HH):
                            feature_maps[w,h,f]=np.sum(self.inputs[w:w + self.Kernels, h:h + self.Kernels, ch]*self.weights[f, ch, :, :])+self.bias[f]
                

        except IndexError:
            print("I m having Index error")
        return feature_maps
    
# This function updates the weights using the momentum
    # params: @dy       : labels
    # return: @xx       : updated weights for each feature
    def backward_propagation(self, dy):

        Width, Height, Channels = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)
        dv = np.zeros(self.weights.shape)
        momentum = 0.9
        Width, Height, F = dy.shape
        for f in range(F):
            for ch in range(Channels):
                for w in range(Width):
                    for h in range(Height):
                        dw[f,ch,:,:]+=dy[w,h,f]*self.inputs[w:w+self.Kernels,h:h+self.Kernels, ch]
                        dx[w:w+self.Kernels,h:h+self.Kernels, ch]+=dy[w,h,f]*self.weights[f,ch,:,:]


        for f in range(F):
            db[f] = np.sum(dy[:, :, f])

        dv = momentum*dv+self.LRate * dw
        self.weights -= dv
        self.bias -= self.LRate * db
        return dx
    
   #extracts weights
    def extract_weights(self):
        return {self.Layer_name +'.weights':self.weights, self.Layer_name +'.bias':self.bias}
    
    #feeding weights
    def feed_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
#This is the class for the Maxpool layer. All the variables related to maxpool
#layer are initialized in this class. When called from the networks.py file, 
#it will be used to generate an instance of the maxpool layer       

class Maxpool_Layer:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.ST = stride
        self.Layer_name = name

#This is the function for forward propagation in the maxpool Layer 
    #params: @inputs            : input for forward propagation
    #return: @out      : matrix of all the max values in the patches

    def forward_propagation(self, inputs):
        try:

            self.inputs = inputs
            Width, Height, Channels = inputs.shape
            new_width = int((Width - self.pool)/self.ST)+ 1
            new_height = int((Height - self.pool)/self.ST) + 1

            
            l1 = int(Width/self.ST)
            l2 = int(Height/self.ST)
 
    
            out = np.zeros(( new_width, new_height, Channels))

            for c in range(Channels):
                for w in range(l1):
                    for h in range(l2):

                        out[ w, h, c] = np.max(self.inputs[ w*self.ST:w*self.ST+self.pool, h*self.ST:h*self.ST+self.pool, c])

        except:
            print("Error in Maxpooling Layer")

        return out
    # This function updates the masking values
    # params: @dy       : labels
    # return: @masking value       

    def backward_propagation(self, dy):
        masking = np.ones_like(self.inputs)*0.25
        return masking*(np.repeat(np.repeat(dy[0:-4, 0:-4 :],2,axis=0),2,axis=1))
    

    def extract_weights(self):
        return 

#This is the class for the Fullyconnected layer. All the variables related to fully connected
#layer are initialized in this class. When called from the networks.py file, 
#it will be used to generate an instance of the fullyconnected layer       
      
class FullyConnected_Layer:

    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.LRate = learning_rate
        self.Layer_name = name
        
    #This is the function for forward propagation in the fully connected Layer 
    #params: @inputs            : input for forward propagation
    #return:  dot product of input and weights

    def forward_propagation(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T
   
    # This function updates the weights using momentum
    # params: @dy       : labels
    # return: @dx       : image data weights

    def backward_propagation(self, dy):

        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)
        dv = np.zeros(self.weights.shape)
        momentum = 0.9

        dv = momentum*dv+self.LRate * dw.T
        self.weights -= dv
        self.bias -= self.LRate * db

        return dx
    
        #extracts weights
    def extract_weights(self):
        return {self.Layer_name +'.weights':self.weights, self.Layer_name +'.bias':self.bias}
    
        #feeds weights
    def feed_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
#This is the class for the flattening layer. All the variables related to flattening
#layer are initialized in this class. When called from the networks.py file, 
#it will be used to generate an instance of the flattening layer          

class Flattening_Layer:
    def __init__(self):
        pass
    
        #This is the function for forward propagation in the flattening Layer 
    #params: @inputs            : input for forward propagation
    #return:  product of channel, width and height
    def forward_propagation(self, inputs):
        self.Width, self.Height, self.Channels, = inputs.shape
        return inputs.reshape(1, self.Channels*self.Width*self.Height)
    
        # This function gives the shape of data
    # params: @dy       : labels
    # return: 1 D dy

    def backward_propagation(self, dy):
        return dy.reshape( self.Width, self.Height, self.Channels)
    
 
    def extract_weights(self):
        return


class LeakyReLuActivation:
    def __init__(self):
        pass
    

    def forward_propagation(self, inputs):
        self.inputs = inputs
        relu = inputs.copy()
        relu[relu < 0] = 0.01*relu[relu < 0]
        return relu
    

    def backward_propagation(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0.01*dx[self.inputs<0]
        return dx
    

    def extract_weights(self):
        return
    

class Softmax_Layer:
    def __init__(self):
        pass

    def forward_propagation(self, inputs):
        exp_prob = np.exp(inputs, dtype=np.float)
        self.out = exp_prob/np.sum(exp_prob)

        return self.out
    

    def backward_propagation(self, dy):
        return self.out.T - dy.reshape(dy.shape[1],1)
    

    def extract_weights(self):
        return

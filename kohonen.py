

import numpy as np
import matplotlib.pylab as plb

def kohonen(eta,size_k,sigma,tmax):
    """Optimizes for either learning rate, network size, nbh parameter, and max iterations.
    
    Args: 
        eta: learning rate
        size_k(int): network size
        sigma:  parameter for nbh function
        tmax(int): Maximum number of iterations.
     
    Plots:
        Error with iterations
        Log-spaces error with iterations
        Derivative of log-spaced error with iterations
    
    Returns:
        None
   
    """
    plb.close('all')
    
    dim = 28*28
    data_range = 255.0
    
    # load in data and labels    
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    # select 4 digits    
    name = 'Arvin Subramaniam' # REPLACE BY YOUR OWN NAME
    targetdigits = name2digits(name) # assign the four digits that should be used
    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    
    dy, dx = data.shape
    
    #initialise the centers randomly
    np.random.seed(123)
    centers = np.random.rand(size_k**2, dim) * data_range
    #print(centers.shape,"shape of centers")
    
    # Initial visualization:
    for i in range(1, size_k**2+1):
        plb.subplot(size_k,size_k,i)
        
        plb.imshow(np.reshape(centers[i-1,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')
    print("Initial centers")    
    plb.show()
    plb.draw()
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    
    #set the maximal iteration count
    #tmax = 10000 # this might or might not work; use your own convergence criterion
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    #print(i_random.shape,"shape of i_random")
    #np.random.shuffle(i_random)
    
    grads = []
    grad_reduced = [] #Logarithmically spaced gradients
    samples = np.geomspace(1,5000,200) #Sample 20 points on a log scale
    for t, i in enumerate(i_random):#And so it goes through all the data sets (2000 points) 10000/2000 = 5 times
        som_step(centers, data[i,:],neighbor,eta,sigma)
        a = som_step(centers, data[i,:],neighbor,eta,sigma)
        grads.append(a)
        
    for i in samples:
        grad_reduced.append(grads[int(i)])
    
    #Plots of gradients vs iterations
    print(len(grads), "length of gradients measured")
    plb.title("Full gradient")
    plb.plot(np.linspace(1,len(grads),len(grads)),grads,'b')
    plb.xlabel("Iterations")
    plb.ylabel("Updates")
    plb.show()
    
    print(len(grad_reduced), "length of spaced gradient measurements")
    plb.title("Reduced gradient")
    plb.plot(np.linspace(1,len(grad_reduced),len(grad_reduced)),grad_reduced,'b')
    plb.xlabel("Iterations")
    plb.ylabel("Updates")
    plb.show()
    
    #Plot derivatives of reduced grads to check
    der = np.gradient(grad_reduced)
    plb.title("Rate of change of gradient")
    plb.plot(np.linspace(1,len(der),len(der)),der,'b')
    plb.xlabel("Iterations")
    plb.ylabel("Rate of change of updates")
    plb.show()


    # Final visualization:
    for i in range(1, size_k**2+1):
        plb.subplot(size_k,size_k,i)
        
        plb.imshow(np.reshape(centers[i-1,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')
        
    # leave the window open at the end of the loop
    print("Final centers")
    plb.show()
    plb.draw()
   
    

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """
    
    size_k = int(np.sqrt(len(centers)))
    
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
        
    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights        
        centers[j,:] += disc * eta * (data - centers[j,:])
        
    grad = np.linalg.norm(disc * eta * (data - centers)) #magnitude of weight update
    return grad
        

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.int(np.mod(s,x.shape[0]))

    return np.sort(x[t,:])


if __name__ == "__main__":
    kohonen()

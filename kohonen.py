"""Python script for Miniproject 1 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb
import scipy as sp

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

dim = 28*28
data_range = 255.0
    
# load in data and labels    
data = np.array(np.loadtxt('data.txt'))
labels = np.loadtxt('labels.txt')

# select 4 digits    
name = 'Ilaria Ricchi' 
targetdigits = name2digits(name) # assign the four digits that should be used
print(targetdigits,"digits of name")
# this selects all data vectors that corresponds to one of the four digits
data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
dy, dx = data.shape
    

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
    
    #This gives 1-to-2000 in 10 consecutive times (since 20000%2000 = 10)
    i_random = np.arange(tmax) % dy
    
    grads = np.zeros(tmax)
    grads_w = np.zeros(tmax)
    grad = np.zeros(tmax/dy) #Array of length 10
    grad_w = np.zeros(tmax/dy)
    std = np.zeros(tmax/dy) #errorbar if needed
 
    for t, i in enumerate(i_random):#i is tmax%2000, t is up to 20000
        som_step(centers, data[i,:],neighbor,eta,sigma)
        a = som_step(centers, data[i,:],neighbor,eta,sigma)
        grads[t] = np.mean(a)#Averaged over all nodes for each tmax
        grads_w[t] = np.max(a)#Winnning node for each tmax
    
    """Get the averaged and winning gradient for each epoch"""
    print("dy is",dy)
    for i,j in enumerate(np.linspace(0,tmax,tmax/dy +1)):
        print(i,j)
        if i == tmax/dy:
            break
        k = int(j)
        grad[i] = np.mean(grads[0:dy+k])#Average for each epoch(spread by dy)
        grad_w[i] = np.mean(grads_w[0:dy+k])
        std[i] = np.std(grads[k:dy+k])
    
     
    """Plot average and winning gradient for each epoch"""
    #grad_tot = np.sum(grad)
    #grad_tot_w = np.sum(grad_w)
    print(len(grad), "length of grad array")
    plb.title("Average (accross dataset) gradient for each iteration")
    plb.plot(np.linspace(1,len(grad),len(grad)),grad,'b',label="averaged gradient")
    #plb.plot(np.linspace(1,len(grad_w),len(grad_w)),grad_w,'r',label="winning gradient")
    #plb.errorbar(np.linspace(1,len(grad),len(grad)),grad,yerr=std)
    plb.xlabel("Iterations")
    plb.ylabel("Average gradient")
    plb.legend()
    #plb.ylim(0,1500)
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
    
    return centers
    
    
def assign_digit(centers):
    """Args:
        centers: final centers after updating

       Returns:
         digits: Digits assigned to each center 
    """
    #idxs = []
    #for d in targetdigits:
        #for i in labels:
            #if i==d:
                #idxs.append(labels.tolist().index(i))
    
    #Create a "minimum data matrix" to compute distances from
    #data_min = np.zeros((len(targetdigits),data.shape[1]))
    #for c,d in enumerate(targetdigits):
        #for j in idxs_:
            #if labels[j] == d:
                #data_min[c,:] = data[j,:]
                
    #Visualization of minimum data matrix
    #len_ = int(np.sqrt(len(targetdigits)))
    #print("Plot of data min")
    #for i in range(1, len(targetdigits)+1):
        #plb.subplot(len_,len_,i)
        #plb.imshow(np.reshape(data_min[i-1,:], [28, 28]),interpolation='bilinear')
        #plb.axis('off')
    #plb.show()
    #plb.draw()
    
    #winners = []
    #For each center pick the closest number
    #for i in range(centers.shape[0]):
        #dists = np.zeros(data_min.shape[0])
        #for d in range(data_min.shape[0]):
            #dists[d] = sp.spatial.distance.euclidean(centers[i,:],data_min[d,:])
        #arg_win = np.argmin(dists)                                              
        #winners.append(arg_win)
        
    winners = []
    #For each center pick the closest number
    for i in range(centers.shape[0]):
        dists = np.zeros(data.shape[0])
        for d in range(data.shape[0]):
            dists[d] = sp.spatial.distance.euclidean(centers[i,:],data[d,:])
        arg_win = np.argmin(dists)                                              
        winners.append(arg_win)
    
    # Assign winning numbers
    output = np.zeros((centers.shape[0],centers.shape[1]))
    for i,j in enumerate(winners):
        output[i,:] = data[j,:]
    
    # Visualization of winners:
    len_ = int(np.sqrt(len(winners)))
    for i in range(1, len(winners)+1):
        plb.subplot(len_,len_,i)
        plb.imshow(np.reshape(output[i-1,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')
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
                         
     Returns:
         grad: update step for each data point
                         
    """
    
    size_k = int(np.sqrt(len(centers)))
    
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
        
    # update all units
    grads = []
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights        
        centers[j,:] += disc * eta * (data - centers[j,:])
        grads.append(np.linalg.norm(disc * (data - centers[j,:])))
        
    return grads
        

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))


if __name__ == "__main__":
    kohonen()
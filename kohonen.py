"""This script contains the function used for the ULNN mini-project 1
   on Kohonen networks.
"""
import numpy as np
import matplotlib.pylab as plb
#import seaborn as sns
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
    

def kohonen(eta,size_k,sigma,tmax, plot = False):
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
        centers: final positions of nodes after iterations
        grads: averaged gradient for each node accross iterations
   
    """
    plb.close('all')
    
    #initialise the centers randomly
    np.random.seed(123)
    centers = np.random.rand(size_k**2, dim) * data_range
    #print(centers.shape,"shape of centers")
    
    if plot:
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
    
    i_random = np.arange(tmax) % dy
    
    grads = np.zeros(tmax)
    
    for t, i in enumerate(i_random):
        som_step(centers, data[i,:],neighbor,eta,sigma)
        a = som_step(centers, data[i,:],neighbor,eta,sigma)
        grads[t] = np.mean(a)#Averaged over all nodes for each tmax

    if plot:
        # Final visualization:
        for i in range(1, size_k**2+1):
            plb.subplot(size_k,size_k,i)
        
            plb.imshow(np.reshape(centers[i-1,:], [28, 28]),interpolation='bilinear')
            plb.axis('off')
        
        # leave the window open at the end of the loop
        print("Final centers")
        plb.show()
        plb.draw()
    
    return centers, grads


def find_convergence(grads, tmax):
    """Get the averaged and winning gradient for each epoch"""
    grad = np.zeros(tmax/dy) #Array of length 10
    for i,j in enumerate(np.linspace(0,tmax,tmax/dy +1)):
        if i == tmax/dy:
            break
        k = int(j)
        grad[i] = np.mean(grads[0:dy+k])#Average for each epoch(spread by dy)

    print(len(grad), "length of grad array")
    plb.figure(3)
    plb.title("Average (accross dataset) gradient for each iteration")
    plb.plot(np.linspace(1,len(grad),len(grad)),grad,'b',label="averaged gradient")
    plb.xlabel("Epochs")
    plb.ylabel("Average gradient")
    plb.legend()
    plb.show()
    
    return None

    
def assign_digit(centers):
    """Args:
        centers: final centers after updating

       Returns:
         digits: Digits assigned to each center 
    """
  
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
    
    return None
  

def inter_distance(centers):
    """Args: Centers
    
       Returns: norm of inter-distance measure
    """
    d = np.zeros((centers.shape[0],centers.shape[1]))
    for i in range(centers.shape[0]):
        for j in range(centers.shape[0]):
            d[i,j] = sp.spatial.distance.euclidean(centers[i,:],centers[j,:])
    l = np.linalg.norm(d)
    return l

def optimize_params(sizes_k, sigmas, eta = 0.3, tmax=20000):
    """ 
    Chooses the optimal number of nodes and sigma based on maximizing the mean inter-cluster 
    separation.
        Args:
            sizes_k: list of numbers of weights to be use
            sigmas: list of numbers of nbh parameters to be used
            eta: learning rate = 0.3
            tmax: max number of iterations=20000
         
        Plots:
            Heatmap of average inter-cluster separation for each of the parameter choices.
            
        Returns:
            k_op: Optimal k from the list above
            sig_op: Optimal sigma from list aboe
    """
    loss = np.zeros((len(sizes_k),len(sigmas)))
    for i,k in enumerate(sizes_k):
        for j,s in enumerate(sigmas):
            c = kohonen(eta,k,s,tmax)[0]
            loss[i,j]=inter_distance(c)
    arg_k, arg_s = np.unravel_index(loss.argmax(),loss.shape)
    print(loss, loss.shape,"loss to be plotted")
    plb.imshow(loss,cmap="hot",interpolation="bilinear")
    plb.title("Heat map of inter-cluster loss function")
    plb.colorbar()
    plb.xticks(np.arange(len(sigmas)),sigmas)
    plb.yticks(np.arange(len(sizes_k)),sizes_k)
    plb.xlabel("sigmas")
    plb.ylabel("Number of neurons")
    plb.show() 
                    
    k_op = sizes_k[arg_k]
    sig_op = sigmas[arg_s]
    return k_op, sig_op


def tesselate(centers):
    """
    Assigns hard clusters to each data point after convergence of the nodes
        Args:
           centers: Nodes afte convergence
    
       Returns:
           Assignment matrix of dimension num_nodes x dy, which are entries of ones and zeros
    """
    assignments = np.zeros((centers.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        arg_y = i
        dist_each_data = [] #Distance array for each data to all clusters
        for j in range(centers.shape[0]):
            dist_each_data.append(sp.spatial.distance.euclidean(centers[j,:],data[i,:]))
        arg_x = np.argmin(dist_each_data)
        assignments[arg_x,arg_y] = 1
    
    return assignments


def assign_counting(centers,vis=False):
    """
    Assigns digit to prototype by determining the modal digit in each cluster. First, a reduced 
    assignment matrix, of dimensions num_nodes x len(targetdigits) is computed. For this, a
    "reduced labels" vector, of length dy is computed.
    
    Once the modal data for each cluster is determined, the picture shown is simply the center 
    of mass of the modal data in the cluster. 
    
        Args:
            centers: centers after convergence
            assigments: Assignment matrix that gives hard clustering (1 for in the cluster, 0 otherwise)
            vis: For visualization of the center of mass of the modal number in each cluster
        
        Plots:
            Histogram of number of data points for each cluster
        
        Returns:
            Center of mass of modal digits in each cluster
    """
    reduced_lbls = []
    for l in labels:
        for t in targetdigits:
            if l == t:
                reduced_lbls.append(l)          
    
    assign = tesselate(centers)
    non_zero = np.count_nonzero(assign)
    #Create "reduced assignment matrix" first, which gives the bins for each digit
    reduced_assign = np.zeros((centers.shape[0],len(targetdigits)))
    for k in range(centers.shape[0]):
        for i in range(len(targetdigits)):
            for j in range(assign.shape[1]):
                if reduced_lbls[j] == targetdigits[i]:
                    reduced_assign[k,i] += assign[k,j]
                    
    
    sum_ = np.sum(reduced_assign)
    
    #Raise exception to check if reduced_assign is correct
    if sum_ != non_zero:
        print("Function error: Number of elements in assign = {} and reduced = {}, are not equal".format(non_zero,sum_))
        raise RuntimeError
    
    k = int(np.sqrt(centers.shape[0]))
    fig, axs = plb.subplots(k, k, sharey='row', tight_layout=True)#Axs here is in kxk array
    axes = []
    for i in range(k):
        for j in range(k):
            axes.append(axs[i,j])
    #print("Counting matrix is",reduced_assign)
    
    for l in range(reduced_assign.shape[0]):
        axes[l].bar(targetdigits,reduced_assign[l,:],align='center')
        
    #Get modal data in each cluster
    modes = []
    for i in range(reduced_assign.shape[0]):
        arg = np.argmax(reduced_assign[i,:])
        modes.append(targetdigits[arg])
        
    #print("Modal data according to cluster is",modes)
    
    # Display the center of mass of the modal clusters
    com_modes = np.zeros((centers.shape[0],centers.shape[1]))
    for i,m in enumerate(modes):
        diff_modes_in_cluster = [] #Collect all the different modal data in cluster
        for j,a in enumerate(assign):
            if reduced_lbls[j] == m:
                diff_modes_in_cluster.append(data[j,:])
        com_mode = np.mean(diff_modes_in_cluster,axis = 0)
        com_modes[i,:] = com_mode
        
    # Visualization of winners:
    if vis:
        len_ = int(np.sqrt(com_modes.shape[0]))
        for i in range(1, com_modes.shape[0]+1):
            plb.subplot(len_,len_,i)
            
            plb.imshow(np.reshape(com_modes[i-1,:], [28, 28]),interpolation='bilinear')
            plb.axis('off')
               
        plb.show()
        plb.draw()
    
    return com_modes

def intra_cluster(com_modes, assignments):
    """
    For each clusters formed, this function computes the average intracluster distance as
    an error measure.
    
    Args: com_modes: center of mass of modal digits in each cluster
          assignments: Assignment matrix
          
    Returns: Average distance between data points and cluster CoM inside the cluster.
    
    """
    
    intracluster_sep = []
    for j in range(com_modes.shape[0]):
        args = np.nonzero(assignments[j,:])
        data_in_cluster = [data[i] for i in args[0]]
        diff = abs(np.linalg.norm(np.mean(data_in_cluster,axis=0) - com_modes[j,:]))
        intracluster_sep.append(diff)
   
    loss = np.nanmean(intracluster_sep) #take the mean whlist ignoring nans
    return loss


def optimize_params_intra(sizes_k, sigmas, eta = 0.3, tmax=20000):
    """ 
    Optimzing the parameters here entail minimizing the intracluster distance.    
        Args:
            sizes_k: list of numbers of weights to be use
            sigmas: list of numbers of nbh parameters to be used
            eta: learning rate = 0.3
            tmax: max number of iterations=20000
         
        Plots:
            Heatmap of average intra-cluster separation for each of the parameter choices.
            
        Returns:
            k_op: Optimal k from the list above
            sig_op: Optimal sigma from list aboe
    """
    loss = np.zeros((len(sizes_k),len(sigmas)))
    for i,k in enumerate(sizes_k):
        for j,s in enumerate(sigmas):
            c = kohonen(eta,k,s,tmax)[0]
            assign = tesselate(c)
            cm = assign_counting(c)
            loss[i,j]=intra_cluster(cm,assign)
    arg_k, arg_s = np.unravel_index(loss.argmin(),loss.shape)
    #print(loss,loss.shape,"loss function to be plotted")
    plb.figure(5)
    plb.imshow(loss,cmap="hot",interpolation='bilinear')
    plb.title("Heat map of intra-cluster loss function")
    plb.colorbar()
    plb.xticks(np.arange(len(sigmas)),sigmas)
    plb.yticks(np.arange(len(sizes_k)),sizes_k)
    plb.xlabel("sigmas")
    plb.ylabel("Number of neurons")
    plb.show() 
                    
    k_op = sizes_k[arg_k]
    sig_op = sigmas[arg_s]
    return k_op, sig_op
              
def joint_optimization(sizes_k, sigmas, eta = 0.3, tmax=20000):
    """ 
    Joint optimization by maximizing inter-cluster distance and minimizing 
    inta-cluster distance. This is done by "naively" picking the argmax of l_inter - 
    l_intra.
        Args:
            sizes_k: list of numbers of weights to be use
            sigmas: list of numbers of nbh parameters to be used
            eta: learning rate = 0.3
            tmax: max number of iterations=20000
         
        Plots:
            Heatmap of average intra-cluster separation for each of the parameter choices.
            
        Returns:
            k_op: Optimal k from the list above
            sig_op: Optimal sigma from list aboe
    """
    loss_inter = np.zeros((len(sizes_k),len(sigmas)))
    loss_intra = np.zeros((len(sizes_k),len(sigmas)))
    for i,k in enumerate(sizes_k):
        for j,s in enumerate(sigmas):
            c = kohonen(eta,k,s,tmax)[0]
            loss_inter[i,j]=inter_distance(c)
            assign = tesselate(c)
            cm = assign_counting(c)
            loss_intra[i,j]=intra_cluster(cm,assign)
    loss_combined = loss_inter - loss_intra
    arg_k, arg_s = np.unravel_index(loss_combined.argmax(),loss_combined.shape)
    plb.figure(7)
    plb.title("Heat map of inter-cluster - intra-cluster loss function")
    plb.imshow(loss_combined,cmap="hot",interpolation="bilinear")
    plb.colorbar()
    plb.xticks(np.arange(len(sigmas)),sigmas)
    plb.yticks(np.arange(len(sizes_k)),sizes_k)
    plb.xlabel("sigmas")
    plb.ylabel("Number of neurons")
    plb.show() 
                    
    k_op = sizes_k[arg_k]
    sig_op = sigmas[arg_s]
    return k_op, sig_op    
       

def kohonen_sigma(rates,tmax=20000,eta=0.3,k=8):
    """
    This function plots the inter-cluster - intra-cluster loss function vs. sigma
    for each annealing rate, and the averaged (over sigma) loss for each rate.
    
       Args: 
        Rates: annealing rate (alpha) of decrease to sigma=1 in tmax/dy epochs
        tmax: Number of iterations = 20000 by default
        k: Number of neurons = 8 by default
              
       Returns: None
    """
    num_epochs = int(tmax/dy)
    #print(num_epochs,"num of epochs")
    sigmas = []#list of arrays
    av_loss = []#Average losses for each rate
    lossess = []
    lower_sig = 1
    for r in rates:
        upper = int(r*num_epochs) + lower_sig
        #print(upper,lower_sig,num_epochs,"stuff in linspace")
        #print(np.linspace(upper,lower_sig,num_epochs), "each sigma array")
        sigmas.append(np.linspace(upper,lower_sig,num_epochs))
    #print(sigmas,"should be array of sigmas")
    for i,s in enumerate(sigmas):#For each array of sigmas
        #print("Rate is",rates[i])
        #print("sigma is",s)
        losses = []
        for j,si in enumerate(s):
            c = kohonen(eta,k,si,tmax)[0]
            loss_inter = inter_distance(c)
            assign = tesselate(c)
            cm = assign_counting(c)
            loss_intra = intra_cluster(cm,assign)
            losses.append(loss_inter - loss_intra)
        #print(losses,type(losses),"losses after appending")
        av_loss.append(np.mean(np.asarray(losses)))
        lossess.append(np.asarray(losses))
    #print(lossess,type(lossess),"length of lossess array")
    #print(lossess[:],"many arrays")
    #print(lossess[0],losses[0].shape[0],"array?")
    plb.figure(6)
    #print(sigmas[0],"first sigmas")
    #print(sigmas[1],"first sigmas")
    #print(sigmas[2],"first sigmas")
    plb.title("Loss vs. sigma function for different rates")
    plb.plot(sigmas[0][::-1],lossess[0],label="alpha = {}".format(rates[0]))
    plb.plot(sigmas[1][::-1],lossess[1],label="alpha = {}".format(rates[1]))
    plb.plot(sigmas[2][::-1],lossess[2],label="alpha = {}".format(rates[2]))
    plb.xlabel("Sigmas")
    plb.ylabel("Inter-cluster - intra-cluster loss")
    plb.legend()
    plb.hold()
    plb.show()
     
    plb.figure(7)
    plb.title("Averaged loss vs. Rates")
    plb.plot(np.linspace(1,len(av_loss),len(av_loss)),av_loss)
    plb.xlabel("Rates")
    plb.ylabel("Loss averaged over sigmas")
    plb.show()  
    
    #Plot the different variations in sigma's for illustration
    for i, list_ in enumerate(sigmas):
        plb.figure(8)
        plb.title("Change of sigma accross epochs for each rate")
        plb.plot(np.linspace(1,len(list_),len(list_)),list_,label="rate of change of sigma is {}".format(rates[i]))
        plb.xlabel("Epochs")
        plb.ylabel("Sigmas")
        plb.legend()
                  
    return None
                 

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
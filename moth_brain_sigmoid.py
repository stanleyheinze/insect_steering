import nengo
import numpy as np

from scipy.special import expit
def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01): # Tom's function
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)

def noisify_weights(W, noise=0.01): # Tom's function
    """Takes a weight matrix and adds some noise on to non-zero values."""
    N = np.random.normal(scale=noise, size=W.shape)
    # Only noisify the connections (positive values in W). Not the zeros.
    N_nonzero = N * W
    return W + N_nonzero


class MothBrainSigmoid(nengo.Network):
    def __init__(self, weight_noise=0.01, sigmoid_noise=0.01, inhib=1):
        super(MothBrainSigmoid, self).__init__()

        self.noise = sigmoid_noise
                
        self.W_FF_PBNf = noisify_weights(np.array([[0, 1], 
                                              [1, 0]]), noise=weight_noise) # flipflop onto contralateral fast PBN
        self.W_PBNf_GIIA = noisify_weights(np.array([[0, -1],
                                                [-1, 0]]), noise=weight_noise) # fast PBN onto contralateral GII-A
        self.W_FF_GIIA = noisify_weights(np.array([[0, 1],
                                              [1, 0]]), noise=weight_noise) # flipflop onto contralateral GII-A
        self.W_in_FF = noisify_weights(np.array([[1, 0],
                                            [0, 1]]), noise=weight_noise) # input onto ipsilateral flipflop
        self.W_in_PBNs = noisify_weights(np.array([[1, 0],
                                              [0, 1]]), noise=weight_noise) # input onto ipsilateral slow PBN
        self.W_PBNs_FF = noisify_weights(np.array([[0, -1],
                                              [-1, 0]]), noise=weight_noise) # slow PBN onto contralateral flipflop
        self.W_GIIA_motor = noisify_weights(np.array([[-inhib, 0],
                                                 [0, -inhib]]), noise=weight_noise) # GII-A onto ipsilateral motor
        self.W_FF_motor = noisify_weights(np.array([[0, 1],
                                               [1, 0]]), noise=weight_noise) # flipflop onto contralateral motor
        self.W_FF_FF = noisify_weights(np.array([[1, 0],
                                            [0, 1]]), noise=weight_noise) # flipflop "memory"
        
        # Initialise all cells
        self.FF_mem = np.array([0.5,0.5]) # flipflop (GI-A) cell: bg freq 20-30 Hz, max freq 80-200 Hz
        #FF_state = np.array([0.5,0.5])
        #motor_all = np.array([0])
        self.out_GIIA = np.array([0.3,0.3]) # GII-A cell: bg freq not given (but low), max freq ~40 Hz
        self.out_PBNs = np.array([0.5,0.5]) # slow PBN cell: bg freq 20-30 Hz, max freq ~100 Hz, decay to bg over at least 20sec
        self.out_PBNf = np.array([0.5,0.5]) # fast PBN cell: bg and max freq unknown
        #PBNs = np.array([0.5,0.5])
        #PBNf = np.array([0.5,0.5])
        #GIIA = np.array([0.3,0.3])        
        
        with self:
            self.inputL = nengo.Node(None, size_in=1)
            self.inputR = nengo.Node(None, size_in=1)
            self.turn = nengo.Node(None, size_in=1)
            self.brain = nengo.Node(self.update, size_in=2)
            nengo.Connection(self.inputL, self.brain[0], synapse=None)
            nengo.Connection(self.inputR, self.brain[1], synapse=None)
            nengo.Connection(self.brain, self.turn, synapse=None)
    def update(self, t, x):
        
        #input_left = np.array(clist[k][0][int(agent_y-1),int(agent_x-1)])
        #input_right = np.array(clist[k][0][int(agent_y-1),int(agent_x+1)])
        
        v = np.vstack((x[0],x[1])).T
        
        # update motor
        #motor = update_ff(np.vstack((input_right,input_left)))
        self.out_PBNs=noisy_sigmoid(np.dot(self.W_in_PBNs,v[0]) + self.out_PBNs,0.5,0,noise=self.noise)
        #PBNs = np.vstack((PBNs,out_PBNs))
        
        # Calculate flipflop neuron activity
        self.input_2_FF = np.dot(self.W_in_FF,v[0])+np.dot(self.W_PBNs_FF,self.out_PBNs)+np.dot(self.W_FF_FF,self.FF_mem)
        self.FF_mem = noisy_sigmoid(self.input_2_FF,2,0,noise=self.noise)
        switch_threshold = 0.5
        # Update memory / current state of FF
        for i in range(len(self.FF_mem)):
            if self.FF_mem[i] > switch_threshold:
                self.FF_mem[i] += 0.1
            else:
                self.FF_mem[i] -= 0.05

        for i in range(len(self.FF_mem)):
            if v[0][i] > switch_threshold and self.input_2_FF[i] > 0.8:
                self.FF_mem[i] -= 0.5
                self.FF_mem[1-i] += 0.5
                                                                
        np.clip(self.FF_mem, 0, 1) # add some decay
        #FF_state = np.vstack((FF_state,FF_mem))
        
        # Calculate fast PBN activity
        out_PBNf = noisy_sigmoid(np.dot(self.W_FF_PBNf,self.FF_mem)+0.5*self.out_PBNf,2,0.5,noise=self.noise)
        #PBNf = np.vstack((PBNf,out_PBNf))
        self.out_PBNf = out_PBNf
        
        # Calculate GII-A activity
        input_2_GIIA = np.dot(self.W_PBNf_GIIA,self.out_PBNf)+np.dot(self.W_FF_GIIA,self.FF_mem)+self.out_GIIA*0.75
        self.out_GIIA = noisy_sigmoid(input_2_GIIA,0.5,0.7,noise=self.noise)
        #GIIA = np.vstack((GIIA,out_GIIA))
        
        # Calculate motor activity
        input_2_motor = np.dot(self.W_FF_motor,self.FF_mem)+np.dot(self.W_GIIA_motor,self.out_GIIA)
        motor = noisy_sigmoid(input_2_motor,1,0.5,noise=self.noise)
        motor = np.clip(-np.diff(input_2_motor),-1,1) # -1 is left, +1 is right turn, 0 is straight walking        

        return motor

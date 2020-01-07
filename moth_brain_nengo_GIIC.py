# Flipflop circuit based on the flipflop steering circuit of the silkworm moth
# (Bombyx mori). This network is based on the following papers:
# * Olberg, 1983: "Pheromone-triggered flip-flopping interneurons in the ventral
# nerve chord of the silkworm moth, Bombyx mori"
# * Kanzaki & Shibuya, 1992: "Long-lasting excitation of protocerebral bilateral
# neurons in the pheromone-processing pathways of the male moth Bombyx mori"
# * Mishima & Kanzaki, 1999: "Physiological and morphological characterization
# of olfactory descending interneurons of the male silkworm moth Bombyx mori"
# * Namiki, Iwabuchi, Pansopha Kono & Kanzaki, 2014: "Information flow through
# neural circuits for pheromone orientation"
# * Kanzaki, Ikeda & Shibuya 1994: "Morphological and physiological properties of
# pheromone-triggered flip-flopping descending interneurons of the male silkworm
# moth Bombyx mori"
#
# The network is implemented in nengo. It consists of 2 pulses that represent the
# input from the two antennae, 4 neuron ensembles on each side of the brain, 
# and 1 ensemble that represents the motor and generates the turn from the 
# combined input from both sides.
# The electrophysiological properties and connectivity of the four different
# cell types will be briefly described in the following:
#
# * Flipflop cell (stateR / stateL in this model): in response to a pheromone
# pulse, this cell flips its activity depending on its previous activity state.
# E.g. if the cell fires at high frequency, the firing frequency will drastically
# decrease in response to a stimulus, and the other way around if the previous
# firing frequency was low. Background firing frequency is ~15-20 Hz, highest
# frequency ~200 Hz. The delay before an increase in firing frequency is ~0.8s,
# ~0.4s before a decrease. This cell is a descending interneuron that originates
# in the ipsilateral lateral accessory lobe (LAL), gives output to the contra-
# lateral LAL, to neck motor neurons in the contralateral subesophageal 
# ganglion (SOG) and presumably to premotor neurons further downstream. The 
# flipflopping activity is assumed to mediate zig-zagging behaviour when e.g.
# following a pheromone plume. 
# See Olberg 1983, Mishima & Kanzaki 1999.
#
# * GII-A descending neuron: this cells responds to a pheromone pulse with brief
# excitation (up to ~40 Hz, delay ~0.4 s) before going back to baseline 
# (~10-20 Hz). This cell is a descending interneuron that originates in the 
# ipsilateral LAL and descends on the ipsilateral side and. It receives input in
# the same area of the LAL in which the flipflop cell (coming from the contra-
# lateral side) has output branches, which is why we assume for the purpose of 
# this model that GII-A receives input from the flipflop cell. 
# See Mishima & Kanzaki 1999.
#
# * PBN inhibitory interneuron: this neuron type projects from one LAL to the
# contralateral LAL and presumably provides recurrent inhibition between the 
# two LALs. There are two types: a slow and a fast type. The slow neurons show 
# long-lasting excitation that has a time-to-peak of several seconds. Resting
# firing frequency is ~2-4 Hz, peak firing frequency is ~6-10 Hz. The fast
# neurons showed a brief burst of excitation that returned to background firing
# frequency after less than 1.5 s after stimulation. From the paper, branching 
# patterns cannot be discerned, therefore the following connections can only be 
# assumed: In our model, the slow PBN receives input directly from the pheromone
# pulse and inhibits the contralateral flipflop cell. The fast PBN receives input
# from the flipflop cell and inhibits the contralateral GII-A DN.
# See Kanzaki 1992
#
# The recurrent inhibition outlined above leads to the characteristic antagonistic
# firing pattern of the two flipflop cell populations: When the left flipflop is
# in its "high" state, the right flipflop will be in "low" state, and the two 
# will flip state when the next stimulus is presented (see Kanzaki, Ikeda & 
# Shibuya, 1994).
#
# Note: Currently, each cell population has 100 neurons by default, which is 
# unrealistic when considering a real insect brain. In another implementation of
# this network, we therefore implement a new neuron class called flipflop, which
# exhibits flip-flopping behaviour using only one cell. However, all other cell 
# ensembles still consist of 100 cells at this point.
#
# The "turn" neuron ensemble integrates the output from flipflop cells and GII-A
# DNs. While either of the flipflop populations is in high state, the turn command
# is 1 or -1 for left or right. The brief excitation of the GII-A DNs inhibits
# the turn population (this has no basis in biological fact and is assumed based
# on the suggested function of the GII-A DNs) which leads to the turn output
# being briefly kept at ~0. This leads to a brief stretch of straight walking in
# between turns. NOTE: This only works if the GII-A DNs are actually inhibitory,
# which should be experimentally tested. If anyone finds a paper that shows immuno-
# stainings with anti-GABA of these cells, I would be very interested in reading 
# that - if the GII-A DNs are GABA-immunoreactive, they are definitely inhibitory.
#
# The output from the turn ensemble can be fed into a motor and drive a robot or 
# simulation.
#
#
# written by A. Adden, 2016-06-16

import nengo
import numpy as np

syn = 0.2 # slow to match paper (0.8s delay on up, 0.4 down)
plant = None #seed for model generation

def flipflop(x): 
    if x[1] > 0.2: #if trigger
        #return -2*x[0] + 1 #return +/-1
        # cheat for dev
        if x[0] > 0.5:
            return -1
        else:
            return 1
    else:
        return 0
        
def stay(x):#cheating for dev
    if x > 0.5:
        return 1
    else:
        return 0

def pulse_(x,step):#generate pulse of random amplitude every step s
    if x % step > step - 0.1:
        #inp = round(np.random.uniform(0.1, 1.0), 2)
        #return inp
        return 1
    else:
        return 0
        

class MothBrainNengo(nengo.Network):
    def __init__(self, noise=0, inhib=3, N=100):
        super(MothBrainNengo, self).__init__()
        with self:
            ### Left side ###
            self.inputL = nengo.Node(None, size_in=1)
            #state[0] holds value, state[1] holds trigger
            self.stateL = nengo.Ensemble(N,2, max_rates = nengo.dists.Uniform(80, 200),
                        intercepts = nengo.dists.Uniform(0,1), seed = plant)
            
            ### Right side ### 
            self.inputR = nengo.Node(None, size_in=1)
            self.stateR = nengo.Ensemble(N,2, max_rates = nengo.dists.Uniform(80, 200),
                        intercepts = nengo.dists.Uniform(0,1), seed = plant)
            
            ### Connections ###
            nengo.Connection(self.inputL, self.stateL[1], synapse=syn/10) #Trigger
            nengo.Connection(self.stateL[0],self.stateL[0], synapse=syn/2, function=stay) #Memory
            nengo.Connection(self.stateL, self.stateL[0], synapse=syn, function=flipflop)#Update Value
            
            nengo.Connection(self.inputR, self.stateR[1], synapse=syn/10) #Trigger
            nengo.Connection(self.stateR[0],self.stateR[0], synapse=syn/2, function=stay) #Memory
            nengo.Connection(self.stateR, self.stateR[0], synapse=syn, function=flipflop)#Update Value
            
            ### Recurrent inhibition ###
            self.PBN_r_slow = nengo.Ensemble(N,1, max_rates = nengo.dists.Uniform(20, 100),
                        intercepts = nengo.dists.Uniform(-.4,0), seed = plant,
                        encoders = [[1]]*N) 
            self.PBN_l_slow = nengo.Ensemble(N,1, max_rates = nengo.dists.Uniform(20, 100),
                        intercepts = nengo.dists.Uniform(-.4,0), seed = plant,
                        encoders = [[1]]*N)
                        
            self.PBN_r_fast = nengo.Ensemble(N,1, max_rates = nengo.dists.Uniform(20, 100),
                        intercepts = nengo.dists.Uniform(-.4,0), seed = plant,
                        encoders = [[1]]*N)
            self.PBN_l_fast = nengo.Ensemble(N,1, max_rates = nengo.dists.Uniform(20, 100),
                        intercepts = nengo.dists.Uniform(-.4,0), seed = plant,
                        encoders = [[1]]*N)
            # Connections
            nengo.Connection(self.PBN_l_slow, self.stateR.neurons, transform = [[-0.5]] * N, synapse=0.5)# slow as in Kanzaki & Shibuya 1992
            nengo.Connection(self.PBN_r_slow, self.stateL.neurons, transform = [[-0.5]] * N, synapse=0.5)
            nengo.Connection(self.inputR, self.PBN_r_slow, synapse=0.1)#was stateL[1]
            nengo.Connection(self.inputL, self.PBN_l_slow, synapse=0.1)#was stateR[1]
            nengo.Connection(self.stateR[0], self.PBN_l_fast, synapse=0.5)
            nengo.Connection(self.stateL[0], self.PBN_r_fast, synapse=0.5)

            #nengo.Connection(PBN_r, PBN_r[0], synapse=0.05, function = lambda x: x[0]*x[1])
            #nengo.Connection(PBN_l, PBN_l[0], synapse=0.05, function = lambda x: x[0]*x[1])


            #GII-C DN - briefly excited neurons used for straight walking *************
            self.giic_l = nengo.Ensemble(N, 1, max_rates = nengo.dists.Uniform(20, 40))
            self.giic_r = nengo.Ensemble(N, 1, max_rates = nengo.dists.Uniform(20, 40))
            #nengo.Connection(self.PBN_r_fast, self.giia_l.neurons, synapse=0.05, transform = [[-3]] * N)
            #nengo.Connection(self.PBN_l_fast, self.giia_r.neurons, synapse=0.05, transform = [[-3]] * N)
            nengo.Connection(self.inputR[0], self.giic_r, synapse = 0.05)
            nengo.Connection(self.inputL[0], self.giic_l, synapse = 0.05)

            
            ### Motor ###
            self.turn = nengo.Ensemble(N, 1, seed=plant) #L = 1, R = -1
            
            nengo.Connection(self.stateL[0], self.turn, transform = 1)
            nengo.Connection(self.stateR[0], self.turn, transform = -1)
            nengo.Connection(self.giic_r, self.turn.neurons, transform = [[-inhib]]*N)
            nengo.Connection(self.giic_l, self.turn.neurons, transform = [[-inhib]]*N)

        if noise > 0:
            for ens in self.all_ensembles:
                ens.noise = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(mean=0, std=noise))

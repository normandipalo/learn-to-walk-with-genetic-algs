from osim.env import RunEnv
import numpy as np
import copy
import pickle

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 0)
sin=np.sin
file_Name = "w_best"

array=np.array

T=4


alpha=0.01
alpha_0=0.01
#TODO: we should exploit the Fourier property for which higher harmonics weights tend to decays as 1/x^n for smooth and continous functions

#I initialize to 0 the weights list, 4 weights for each muscle (I compose the periodic function with 4 elements of a Fourier Series)
#I define weights only for 9 periodic functions, as I assume that the legs move symmetrically in time.

w=[]

for i in range(9):
    w.append(np.array([0.,0.,0.,0.,0.,0.,0.,0.]))



def output(a,T,t):
    # Output of a 4th degree Fourier Series of sin.
    # INPUT: the 4 harmonics weights, time period T, and the time t.
        y=0
        
        for i in range(4):
            y+=a[i]*sin((i+1)*np.pi*2*t/T+a[i+4])
        return y


def evolve(w):
    #This functions evolves randomly w generating a direction, sampling from gaussians distribution.
    #It operates directly on w so it doesn't return anything.

    for i in range(9):
        w[i]+=np.random.randn(8)*alpha
    """
    for i in range(9):
        delta=[]
        for j in range(4):
            delta.append(np.random.randn()*0.5/((j+1)**2))
        delta=np.asarray(delta)
        w[i]+=delta
    """


def input(w,t):
    #This function generates the input to the model.
    #INPUT: weights, time step
    #Returns: input arrat
    global T
    inputs=[]
    
    #The model input has dimension 18. Eventhough, I only use 9 functions, since the last 9 are just the same but with a -
    #When one of the inputs is <0 it is =0 for the model.
    """
    inputs=[-output(w[0],T,t),output(w[1],T,t),-output(w[2],T,t), output(w[3],T,t),output(w[4],T,t),output(w[5],T,t),-output(w[6],T,t),-output(w[7],T,t),output(w[8],T,t),
    output(w[0],T,t),-output(w[1],T,t),output(w[2],T,t), -output(w[3],T,t), -output(w[4],T,t), -output(w[5],T,t), output(w[6],T,t), output(w[7],T,t), -output(w[8],T,t),]
    """
    inputs=[-output(w[0],T,t),output(w[1],T,t),-output(w[2],T,t), 
    output(w[3],T,t),output(w[4],T,t),output(w[5],T,t),
    -output(w[6],T,t),-output(w[7],T,t),output(w[8],T,t),
    output(w[0],T,t+T/2),-output(w[1],T,t+T/2),output(w[2],T,t+T/2), 
    -output(w[3],T,t+T/2), -output(w[4],T,t+T/2), -output(w[5],T,t+T/2),
     output(w[6],T,t+T/2), output(w[7],T,t+T/2), -output(w[8],T,t+T/2),]

    return inputs





"""inputs=[]
for i in range(400):
    inputs.append([-np.sin(i*np.pi/200)*0.2, np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,
        np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,
        -np.sin(i*np.pi/200)*0.2*1.2,-np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2*1.2,np.sin(i*np.pi/200)*0.2,-0.2*np.sin(i*np.pi/200)])
"""


#############MAIN################

#Initialize the data structures that will be identical to w.
#best_w will be the best performing weights.
#w_try will be the current try


try:
    fileObject = open(file_Name,'r')
    w_best = pickle.load(fileObject)
    fileObject.close()
    print("Best loaded!")
except:
    #w_best=copy.deepcopy(w)
    w_best=([array([ -7.26506485e-01,  -8.38614787e-01,  -5.56358174e-01,
        -6.75451124e-01,   3.22806188e-02,  -5.15739387e-04,
         1.63209237e-01,  -1.91754996e-01]), array([-1.49402493,  1.16777911,  0.07246947, -0.21257161, -0.3912605 ,
        0.55184244, -0.67970783, -0.5405781 ]), array([ 0.0595232 , -0.61581316, -0.62864344, -1.16207122,  0.45738423,
        0.52816882,  0.22419772, -0.00324515]), array([ 0.17626497, -0.35013157,  0.7579556 , -0.32839809,  0.17016797,
       -0.47315695, -0.27232134, -0.1938231 ]), array([-0.11449376,  0.49809987,  1.36149742,  0.74359823,  0.56645939,
        0.96690231, -0.14766671,  0.30231231]), array([-0.59583676,  0.05403124, -1.01776343, -0.5616875 ,  0.45425483,
        0.22633419,  0.18381335, -0.65777158]), array([-0.02919157,  0.4980824 ,  0.64461859, -0.29464231,  0.33684163,
       -0.76444424, -0.65709448, -1.31262128]), array([ 0.09118833, -0.14873715,  0.14209483, -0.80052937,  0.45015313,
       -1.64013086,  0.21587223,  0.00546647]), array([ 0.55469414,  0.46425008, -0.28322548, -0.70177578, -0.53654581,
       -0.07625724,  0.59847321, -0.98369352])])

    w_first=([array([ 0.28032398, -0.22280502, -0.45912563, -0.13087411,  0.00131868,
       -0.22360605,  0.29960409, -0.34270209]), array([ 0.15453372, -0.88020378, -0.06625429, -0.38944187, -0.4778913 ,
        0.10231507, -0.7354027 , -0.38999563]), array([-0.61151511,  0.08259072, -0.11728766, -0.78825239,  0.07503682,
        0.3780213 ,  1.21214736, -0.34271815]), array([ 0.23661985,  0.2680493 ,  0.21691186, -0.48925347,  0.03141434,
        0.05827253,  0.38333856,  0.12633454]), array([-0.60085352, -0.47220083,  0.53696127,  0.2804884 , -0.27651353,
       -0.14954568, -0.29866158, -0.21254649]), array([-0.26184373, -0.2953516 ,  0.51976833,  0.05279496,  0.53992931,
        0.37088401,  0.07927318, -0.27730838]), array([-0.34321139, -0.23739512,  0.15111555,  0.32106653,  0.15391846,
        0.99346677,  0.34962222,  1.03419399]), array([-0.68081062, -0.13637173,  0.65471814, -0.2885456 , -0.77494225,
       -0.2633149 ,  0.05738678,  1.0264855 ]), array([ 0.51705584,  0.40825768,  0.45069928, -0.18134995,  0.52601042,
        1.03569832, -0.19178778,  0.15754662])])
    
    print("Initializing new best")

w_try=copy.deepcopy(w)
best_reward=0.
runs=500
unev_runs=0

print("Baseline, run with w_best")
observation = env.reset(difficulty = 0)
total_reward = 0.0
for i in range(500):
    i*=0.01
    if i>2:
        i-=2
        observation, reward, done, info = env.step(input(w_best,i))
        T=2
    else:
    # make a step given by the controller and record the state and the reward
        observation, reward, done, info = env.step(input(w_first,i))
    total_reward += reward
    if done:
        break
best_reward=total_reward

# Your reward is
print("Total reward %f" % total_reward)


for run in range(runs):

    T=4
    #if it doens't get better for more than 10 iterations, increase alpha to allow bigger changes
    #Increase alpha then set unev_runs back to 0
    if unev_runs>30:
        print("Augmenting alpha")
        alpha+=alpha_0
        unev_runs=0


    unev_runs+=1
    print("Run {}/{}".format(run,runs))
    observation = env.reset(difficulty = 0)

    #I copy the best performing w and I try to evolve it
    w_try=copy.deepcopy(w_best)
    evolve(w_try)

    total_reward = 0.0
    for i in range(500):
        # make a step given by the controller and record the state and the reward
        i*=0.01 #Every step is 0.01 s
        if i>2:
            T=2
            i-=2
            observation, reward, done, info = env.step(input(w_try,i))
        else:
    # make a step given by the controller and record the state and the reward
    
            observation, reward, done, info = env.step(input(w_first,i))
        total_reward += reward
        if done:
            print("done")
            break

    if total_reward>best_reward:
        #If the total reward is the best one, I store w_try as w_best, dump it with pickle and save the reward
        print("Found a better one!")
        unev_runs=0
        alpha=alpha_0
        w_best=copy.deepcopy(w_try)
        print(w_best)

        
        fileObject = open(file_Name,'wb')
        #pickle.dump(w_best,fileObject)
        fileObject.close()
        best_reward=total_reward

        

    # Your reward is
    print("Total reward %f" % total_reward)



#Final run with video and best weights. The raw_input waits for the user to type something. (if it's afk)
print("Run with best weights")
_=raw_input("ready? ")
env = RunEnv(visualize=True)
observation = env.reset(difficulty = 0)


T=4
total_reward = 0.0
for i in range(500):
    # make a step given by the controller and record the state and the reward
    i*=0.01 #Every step is 0.01 s

    if i>2:
        i-=2
        T=2
        observation, reward, done, info = env.step(input(w_best,i))
    else:
    # make a step given by the controller and record the state and the reward
    
        observation, reward, done, info = env.step(input(w_first,i))
    total_reward += reward
    if done:
        print("done")
        break
    

# Your reward is
print("Total reward %f" % total_reward)

print("best weights")
print(w_best)

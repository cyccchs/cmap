import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
import logging
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from keras.datasets import cifar10
import matplotlib.patches as patches
import csv
logging.getLogger().setLevel(logging.INFO)

def sample_normal_single(mean, stddev, name=None):
	return tf.random_normal(
		# shape=mean.get_shape(),
		shape=tf.shape(mean),
    	mean=mean,
    	stddev=stddev,
    	dtype=tf.float32,
    	seed=None,
    	name=name,
		)
    
def alpha_reward(alpha, reward):
    rewards1, rewards2, rewards3, rewards4 = [], [], [], []
    print("sssssssssssss")
    
    for i in range(160):
        rewards1.append(4*reward[i]*alpha[i][0] - tf.cast(tf.equal(reward[i], 0), tf.float32)*alpha[i][0]*4)
        rewards2.append(4*reward[i]*alpha[i][1] - tf.cast(tf.equal(reward[i], 0), tf.float32)*alpha[i][1]*4)
        rewards3.append(4*reward[i]*alpha[i][2] - tf.cast(tf.equal(reward[i], 0), tf.float32)*alpha[i][2]*4)
        rewards4.append(4*reward[i]*alpha[i][3] - tf.cast(tf.equal(reward[i], 0), tf.float32)*alpha[i][3]*4)
    

    return rewards1, rewards2, rewards3, rewards4

def plot_images_labels_prediction_box_printer(images,labels,prediction,loc1,loc2,loc3,loc4,glimpse_size,num=16):
    
    
    if num>25: num=25
    for j in range(0,len(loc1)):
        fig = plt.gcf()
        fig.suptitle('Glimpse:'+str(j))
        fig.set_size_inches(20,5)
        idx = 0
        for i in range(0,num):
            print(i)
            
            ax=plt.subplot(2,8,1+i)
            
            ax.imshow(np.reshape(images[idx],(32,32,3)))
            title="label=" +str(labels[idx])
            if len(prediction)>0:
                title+=",predict="+str(prediction[idx])
            rect1 = patches.Rectangle((abs(16*loc1[j][i][1]-16)-glimpse_size/2,16*loc1[j][i][0]+16-glimpse_size/2),glimpse_size,glimpse_size,linewidth=2,edgecolor='r',facecolor='none')   
            rect2 = patches.Rectangle((abs(16*loc2[j][i][1]-16)-glimpse_size/2,16*loc2[j][i][0]+16-glimpse_size/2),glimpse_size,glimpse_size,linewidth=2,edgecolor='b',facecolor='none') 
            rect3 = patches.Rectangle((abs(16*loc3[j][i][1]-16)-glimpse_size/2,16*loc3[j][i][0]+16-glimpse_size/2),glimpse_size,glimpse_size,linewidth=2,edgecolor='k',facecolor='none') 
            rect4 = patches.Rectangle((abs(16*loc4[j][i][1]-16)-glimpse_size/2,16*loc4[j][i][0]+16-glimpse_size/2),glimpse_size,glimpse_size,linewidth=2,edgecolor='m',facecolor='none')  
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.add_patch(rect3)
            ax.add_patch(rect4)
            ax.set_title(title,fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
            idx+=1
        #plt.savefig('C:\\Users\\WesKao\\.spyder-py3\\multi agent 4 cifar stop ht ver2 selfatt new locnet\\saver 1 glimpse stop gt lr4\\test'+ str(j) +'.jpg')
        plt.show()
        
def conv2d(x,w, stride):
	return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

def EVALUATION(PREDICT, LABEL, F_belta, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, flag):
    if flag==0:
        for i in range(len(PREDICT)):
            if (PREDICT[i]==0 or PREDICT[i]==1) and (LABEL[i]==0 or LABEL[i]==1):
                TP1+=1
                if PREDICT[i]==0 and LABEL[i]==0:
                    TP2+=1
                elif PREDICT[i]==1 and LABEL[i]==1:
                    TN2+=1
                elif PREDICT[i]==1 and LABEL[i]==0:
                    FN2+=1
                elif PREDICT[i]==0 and LABEL[i]==1:
                    FP2+=1
            elif PREDICT[i]==2 and LABEL[i]==2:
                TN1+=1
            elif (PREDICT[i]==0 or PREDICT[i]==1) and LABEL[i]==2:
                FP1+=1
            elif PREDICT[i]==2 and (LABEL[i]==0 or LABEL[i]==1):
                FN1+=1
        return TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2
    else:
        print("TP1 =",TP1,"TN1 =", TN1,"FP1 =", FP1,"FN1 =", FN1, "|","TP2 =", TP2,"TN2 =", TN2,"FP2 =", FP2,"FN2 =", FN2)
        try:
            PRECISION1 = TP1/(TP1+FP1)
            PRECISION2 = TP2/(TP2+FP2)
        except:
            PRECISION1 = 0
            PRECISION2 = 0
        try:
            RECALL1 = TP1/(TP1+FN1)
            RECALL2 = TP2/(TP2+FN2)
        except:
            RECALL1 = 0
            RECALL2 = 0
        acc1 = (TP1+TN1)/(TP1+TN1+FP1+FN1)
        try:
            acc2 = (TP2+TN2)/(TP2+TN2+FP2+FN2)
        except:
            acc2 = 0
        try:
            F_score1 = ((1+F_belta*F_belta)*PRECISION1*RECALL1)/(F_belta*F_belta*PRECISION1+RECALL1)
            F_score2 = ((1+F_belta*F_belta)*PRECISION2*RECALL2)/(F_belta*F_belta*PRECISION2+RECALL2)
        except:
            F_score1 = 0
            F_score2 = 0
#        print(TP,TN,FP,FN)
#        print('PRECISION = ', PRECISION)
#        print('RECALL = ', RECALL)
#        print('|ACCURACY| = ', round(acc,2), '|F_score| = ', round(F_score,2))
        return PRECISION1, RECALL1, PRECISION2, RECALL2,  acc1, acc2, F_score1, F_score2
""" 
def rnn_decoder123(rnn_out, decoder_inputs, initial_state,cell, loop_function=None, soft_attention=None, scope=None):
    #with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):#inp:[160 256]就是gt
            if loop_function is not None and prev is not None:#第一次不會進
                #with variable_scope.variable_scope("loop_function", reuse=True):
                glimpse_unstack = loop_function(prev, i)
                
                print('rnn1')
            if i >= 0:#soft attention maybe added in here (without CNN) 第一次沒有loopfunction因為已在外先做了
                #hard的lstm取代soft的lstm
                #variable_scope.get_variable_scope().reuse_variables()
                if i == 0:
                    prev = rnn_out
                    glimpse_unstack = inp
                alpha, z,z_list,m_concat = soft_attention(prev, glimpse_unstack) #alpha:[1 256] z:[1 512]
                rnn_out, state = cell(z, state)#ht:[160 256]
                outputs.append(rnn_out)
                
                print('rnn2')
            if loop_function is not None:
                prev = rnn_out
                print('rnn3')
            print('rnn4')
        print('rnn5')
        return outputs,state,inp
"""   
    
    
    
def plot_images_labels_prediction(images,labels,prediction,idx,num=16):
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(2,8,1+i)
        
        ax.imshow(np.reshape(images[idx],(32,32,3)))
        title="label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()


def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def _log_likelihood(mean_ts, loc_ts, std):
    """
    Args
    - mean_ts: a list with len=num_glimpses contains tensors with shape (B, 2)
    - loc_ts: a list with len=num_glimpses contains tensors with shape (B, 2), sampled location of all timesteps
    - std: scalar
    Returns
    - logll: tensor with shape (B, timesteps)
    """
    mean_ts = tf.stack(mean_ts)  # [timesteps, batch_sz, loc_dim]
    loc_ts = tf.stack(loc_ts)
    gaussian = Normal(mean_ts, std)
    logll = gaussian._log_prob(x=loc_ts)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)  # reduce location(dim=2) to 1
    return tf.transpose(logll)      # [batch_sz, timesteps]



class RetinaSensor1(object):
    """
    A retina that extracts a `patch` around location `loc_t` from image `img_ph`.
    Args
    ----
    - img_ph: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    - loc_t: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
    - pth_size: a scalar. Size of the square glimpse patch.
    Returns
    -------
    - patch: a 4D tensor of shape (B, pth_size, pth_size, 1). The foveated glimpse of the image.
    """
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc_t):
        img = tf.reshape(img_ph, [tf.shape(img_ph)[0], self.img_size, self.img_size, 3])
        pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc_t)
        pth = tf.reshape(pth, [tf.shape(loc_t)[0], self.pth_size*self.pth_size*3])
        return pth

class RetinaSensor2(object):
    """
    A retina that extracts a `patch` around location `loc_t` from image `img_ph`.
    Args
    ----
    - img_ph: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    - loc_t: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
    - pth_size: a scalar. Size of the square glimpse patch.
    Returns
    -------
    - patch: a 4D tensor of shape (B, pth_size, pth_size, 1). The foveated glimpse of the image.
    """
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc_t):
        img = tf.reshape(img_ph, [tf.shape(img_ph)[0], self.img_size, self.img_size, 3])
        pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc_t)
        pth = tf.reshape(pth, [tf.shape(loc_t)[0], self.pth_size*self.pth_size*3])
        return pth

class RetinaSensor3(object):
    """
    A retina that extracts a `patch` around location `loc_t` from image `img_ph`.
    Args
    ----
    - img_ph: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    - loc_t: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
    - pth_size: a scalar. Size of the square glimpse patch.
    Returns
    -------
    - patch: a 4D tensor of shape (B, pth_size, pth_size, 1). The foveated glimpse of the image.
    """
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc_t):
        img = tf.reshape(img_ph, [tf.shape(img_ph)[0], self.img_size, self.img_size, 3])
        pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc_t)
        pth = tf.reshape(pth, [tf.shape(loc_t)[0], self.pth_size*self.pth_size*3])
        return pth
    
class RetinaSensor4(object):
    """
    A retina that extracts a `patch` around location `loc_t` from image `img_ph`.
    Args
    ----
    - img_ph: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    - loc_t: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
    - pth_size: a scalar. Size of the square glimpse patch.
    Returns
    -------
    - patch: a 4D tensor of shape (B, pth_size, pth_size, 1). The foveated glimpse of the image.
    """
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc_t):
        img = tf.reshape(img_ph, [tf.shape(img_ph)[0], self.img_size, self.img_size, 3])
        pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc_t)
        pth = tf.reshape(pth, [tf.shape(loc_t)[0], self.pth_size*self.pth_size*3])
        return pth

class LocationNetwork1(object):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    paself.trize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - hidden_size: rnn hidden size
    - loc_dim: location dim = 2
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for the current time step `t`.
    Returns
    -------
    - mean: a 2D vector of shape (B, 2). Gaussian mean of current time step.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    """
    def __init__(self, hidden_size, loc_dim, std=0.22, is_sampling=True):
        self.loc_dim = loc_dim
        self.std = std
        self.w = _weight_variable((hidden_size, loc_dim))
        self.b = _bias_variable((loc_dim,))
        self.is_sampling = is_sampling


    def __call__(self, h_t):
        # compute mean at this time step
        mean_t = tf.nn.xw_plus_b(tf.stop_gradient(h_t), self.w, self.b)
        mean_t = tf.clip_by_value(mean_t, -1., 1.)

        def is_sampling_true():
            # sample from gaussian parameterized by this mean when training
            loc_t = tf.stop_gradient(sample_normal_single(mean_t, stddev=self.std))
            #loc_t = tf.clip_by_value(loc_t, -1., 1.)
            return loc_t

        def is_sampling_false():
            # using mean when testing
            return  mean_t

        loc_t = tf.cond(self.is_sampling, is_sampling_true, is_sampling_false)

        #loc_t = tf.stop_gradient(loc_t)

        return loc_t, mean_t

class LocationNetwork2(object):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    paself.trize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - hidden_size: rnn hidden size
    - loc_dim: location dim = 2
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for the current time step `t`.
    Returns
    -------
    - mean: a 2D vector of shape (B, 2). Gaussian mean of current time step.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    """
    def __init__(self, hidden_size, loc_dim, std=0.22, is_sampling=True):
        self.loc_dim = loc_dim
        self.std = std
        self.w = _weight_variable((hidden_size, loc_dim))
        self.b = _bias_variable((loc_dim,))
        self.is_sampling = is_sampling


    def __call__(self, h_t):
        # compute mean at this time step
        mean_t = tf.nn.xw_plus_b(tf.stop_gradient(h_t), self.w, self.b)
        mean_t = tf.clip_by_value(mean_t, -1., 1.)

        def is_sampling_true():
            # sample from gaussian parameterized by this mean when training
            loc_t = tf.stop_gradient(sample_normal_single(mean_t, stddev=self.std))
            #loc_t = tf.clip_by_value(loc_t, -1., 1.)
            return loc_t

        def is_sampling_false():
            # using mean when testing
            return  mean_t

        loc_t = tf.cond(self.is_sampling, is_sampling_true, is_sampling_false)

        #loc_t = tf.stop_gradient(loc_t)

        return loc_t, mean_t
    
class LocationNetwork3(object):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    paself.trize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - hidden_size: rnn hidden size
    - loc_dim: location dim = 2
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for the current time step `t`.
    Returns
    -------
    - mean: a 2D vector of shape (B, 2). Gaussian mean of current time step.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    """
    def __init__(self, hidden_size, loc_dim, std=0.22, is_sampling=True):
        self.loc_dim = loc_dim
        self.std = std
        self.w = _weight_variable((hidden_size, loc_dim))
        self.b = _bias_variable((loc_dim,))
        self.is_sampling = is_sampling


    def __call__(self, h_t):
        # compute mean at this time step
        mean_t = tf.nn.xw_plus_b(tf.stop_gradient(h_t), self.w, self.b)
        mean_t = tf.clip_by_value(mean_t, -1., 1.)

        def is_sampling_true():
            # sample from gaussian parameterized by this mean when training
            loc_t = tf.stop_gradient(sample_normal_single(mean_t, stddev=self.std))
            #loc_t = tf.clip_by_value(loc_t, -1., 1.)
            return loc_t

        def is_sampling_false():
            # using mean when testing
            return  mean_t

        loc_t = tf.cond(self.is_sampling, is_sampling_true, is_sampling_false)

        #loc_t = tf.stop_gradient(loc_t)

        return loc_t, mean_t
    
class LocationNetwork4(object):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    paself.trize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - hidden_size: rnn hidden size
    - loc_dim: location dim = 2
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for the current time step `t`.
    Returns
    -------
    - mean: a 2D vector of shape (B, 2). Gaussian mean of current time step.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    """
    def __init__(self, hidden_size, loc_dim, std=0.22, is_sampling=True):
        self.loc_dim = loc_dim
        self.std = std
        self.w = _weight_variable((hidden_size, loc_dim))
        self.b = _bias_variable((loc_dim,))
        self.is_sampling = is_sampling


    def __call__(self, h_t):
        # compute mean at this time step
        mean_t = tf.nn.xw_plus_b(tf.stop_gradient(h_t), self.w, self.b)
        mean_t = tf.clip_by_value(mean_t, -1., 1.)

        def is_sampling_true():
            # sample from gaussian parameterized by this mean when training
            loc_t = tf.stop_gradient(sample_normal_single(mean_t, stddev=self.std))
            #loc_t = tf.clip_by_value(loc_t, -1., 1.)
            return loc_t

        def is_sampling_false():
            # using mean when testing
            return  mean_t

        loc_t = tf.cond(self.is_sampling, is_sampling_true, is_sampling_false)

        #loc_t = tf.stop_gradient(loc_t)

        return loc_t, mean_t

class GlimpseNetwork1(object):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `pth_t` to
    a fc layer and the glimpse location vector `loc_t`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `glimpse_t = relu( fc( fc(pth_t) ) + fc( fc(loc_t) ) )`
    Args
    ----
    - pth_size: pth size
    - loc_dim: location dim = 2
    - g_size: hidden layer size of the fc layer for `pths`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - output_size: output size of this network.
    - pth_t: a 4D Tensor of shape (B, pth_size, pth_size, 1). Current time step minibatch of pths.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    Returns
    -------
    - glimpse_t: a 2D tensor of shape (B, output_size). The glimpse representation returned by the glimpse network for the current timestep `t`.
    """
    def __init__(self, pth_size, loc_dim, g_size, l_size, output_size):
        # layer 1
        self.g1_w = _weight_variable((pth_size*pth_size*3, g_size))
        self.g1_b = _bias_variable((g_size,))

        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))

        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))

        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, pth_t, loc_t):
        # feed pths and locs to respective fc layers
        what  = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pth_t, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        where = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(loc_t, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        # feed to fc layer
        glimpse_t = tf.nn.relu(what + where)
        return glimpse_t

class GlimpseNetwork2(object):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `pth_t` to
    a fc layer and the glimpse location vector `loc_t`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `glimpse_t = relu( fc( fc(pth_t) ) + fc( fc(loc_t) ) )`
    Args
    ----
    - pth_size: pth size
    - loc_dim: location dim = 2
    - g_size: hidden layer size of the fc layer for `pths`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - output_size: output size of this network.
    - pth_t: a 4D Tensor of shape (B, pth_size, pth_size, 1). Current time step minibatch of pths.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    Returns
    -------
    - glimpse_t: a 2D tensor of shape (B, output_size). The glimpse representation returned by the glimpse network for the current timestep `t`.
    """
    def __init__(self, pth_size, loc_dim, g_size, l_size, output_size):
        # layer 1
        self.g1_w = _weight_variable((pth_size*pth_size*3, g_size))
        self.g1_b = _bias_variable((g_size,))

        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))

        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))

        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, pth_t, loc_t):
        # feed pths and locs to respective fc layers
        what  = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pth_t, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        where = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(loc_t, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        # feed to fc layer
        glimpse_t = tf.nn.relu(what + where)
        return glimpse_t
    
class GlimpseNetwork3(object):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `pth_t` to
    a fc layer and the glimpse location vector `loc_t`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `glimpse_t = relu( fc( fc(pth_t) ) + fc( fc(loc_t) ) )`
    Args
    ----
    - pth_size: pth size
    - loc_dim: location dim = 2
    - g_size: hidden layer size of the fc layer for `pths`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - output_size: output size of this network.
    - pth_t: a 4D Tensor of shape (B, pth_size, pth_size, 1). Current time step minibatch of pths.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    Returns
    -------
    - glimpse_t: a 2D tensor of shape (B, output_size). The glimpse representation returned by the glimpse network for the current timestep `t`.
    """
    def __init__(self, pth_size, loc_dim, g_size, l_size, output_size):
        # layer 1
        self.g1_w = _weight_variable((pth_size*pth_size*3, g_size))
        self.g1_b = _bias_variable((g_size,))

        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))

        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))

        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, pth_t, loc_t):
        # feed pths and locs to respective fc layers
        what  = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pth_t, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        where = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(loc_t, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        # feed to fc layer
        glimpse_t = tf.nn.relu(what + where)
        return glimpse_t
    
class GlimpseNetwork4(object):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `pth_t` to
    a fc layer and the glimpse location vector `loc_t`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `glimpse_t = relu( fc( fc(pth_t) ) + fc( fc(loc_t) ) )`
    Args
    ----
    - pth_size: pth size
    - loc_dim: location dim = 2
    - g_size: hidden layer size of the fc layer for `pths`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - output_size: output size of this network.
    - pth_t: a 4D Tensor of shape (B, pth_size, pth_size, 1). Current time step minibatch of pths.
    - loc_t: a 2D vector of shape (B, 2). Current time step location sampled from guassian(mean)
    Returns
    -------
    - glimpse_t: a 2D tensor of shape (B, output_size). The glimpse representation returned by the glimpse network for the current timestep `t`.
    """
    def __init__(self, pth_size, loc_dim, g_size, l_size, output_size):
        # layer 1
        self.g1_w = _weight_variable((pth_size*pth_size*3, g_size))
        self.g1_b = _bias_variable((g_size,))

        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))

        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))

        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, pth_t, loc_t):
        # feed pths and locs to respective fc layers
        what  = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pth_t, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        where = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(loc_t, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        # feed to fc layer
        glimpse_t = tf.nn.relu(what + where)
        return glimpse_t

class BaseLineNetwork1(object):
    """
    Regresses the baseline in the reward function to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_ts: the hidden state vectors of the core network for the all time step `ts`.
    Returns
    -------
    - baselines: a 2D vector of shape (B, timesteps). The baseline for the all time step `ts`.
    """

    def __init__(self, hidden_size):
        self.w = _weight_variable((hidden_size, 1))
        self.b = _bias_variable((1, ))

    def __call__(self, h_ts):
        # Time independent baselines
        baselines = []
        for h_t in h_ts[1:]:
            baseline = tf.nn.xw_plus_b(h_t, self.w, self.b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)

        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        return baselines
    
class BaseLineNetwork2(object):
    """
    Regresses the baseline in the reward function to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_ts: the hidden state vectors of the core network for the all time step `ts`.
    Returns
    -------
    - baselines: a 2D vector of shape (B, timesteps). The baseline for the all time step `ts`.
    """

    def __init__(self, hidden_size):
        self.w = _weight_variable((hidden_size, 1))
        self.b = _bias_variable((1, ))

    def __call__(self, h_ts):
        # Time independent baselines
        baselines = []
        for h_t in h_ts[1:]:
            baseline = tf.nn.xw_plus_b(h_t, self.w, self.b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)

        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        return baselines
    
class BaseLineNetwork3(object):
    """
    Regresses the baseline in the reward function to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_ts: the hidden state vectors of the core network for the all time step `ts`.
    Returns
    -------
    - baselines: a 2D vector of shape (B, timesteps). The baseline for the all time step `ts`.
    """

    def __init__(self, hidden_size):
        self.w = _weight_variable((hidden_size, 1))
        self.b = _bias_variable((1, ))

    def __call__(self, h_ts):
        # Time independent baselines
        baselines = []
        for h_t in h_ts[1:]:
            baseline = tf.nn.xw_plus_b(h_t, self.w, self.b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)

        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        return baselines
    
class BaseLineNetwork4(object):
    """
    Regresses the baseline in the reward function to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_ts: the hidden state vectors of the core network for the all time step `ts`.
    Returns
    -------
    - baselines: a 2D vector of shape (B, timesteps). The baseline for the all time step `ts`.
    """

    def __init__(self, hidden_size):
        self.w = _weight_variable((hidden_size, 1))
        self.b = _bias_variable((1, ))

    def __call__(self, h_ts):
        # Time independent baselines
        baselines = []
        for h_t in h_ts[1:]:
            baseline = tf.nn.xw_plus_b(h_t, self.w, self.b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)

        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        return baselines

class Selfattention(object):
    def __init__(self):
        self.W = 256
        self.Wq = 256
        self.Wk = 256
        self.Wv = 256
    def __call__(self,g1, g2, g3, g4):
        a = tf.stop_gradient(tf.stack([g1, g2, g3, g4], 1))
        x = tf.layers.dense(a, self.W)
        print(x)
        q = tf.layers.dense(x, self.Wq)#[? 4 256]
        k = tf.layers.dense(x, self.Wk)#[? 4 256]
        v = tf.layers.dense(x, self.Wv)#[? 4 256]
        print("llllllllll")
        q_trans = tf.transpose(q, [0, 2, 1])#[? 256 4]
        print(q_trans)
        alpha = tf.nn.softmax(tf.matmul(k, q_trans)/self.W**0.5)#(?, 4, 4)
        print(alpha)
        b = tf.matmul(alpha, v)#[? 4 256]
        print(b)
        b = tf.unstack(b, axis = 1)
        print(b)
        return b

class ClassificationNetwork(object):
    """
    Uses the internal state `h_last` of the core network to
    produce the final output classification.
    Concretely, feeds the hidden state `h_last` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.
    the network is simply a linear softmax classifier.
    Args
    ----
    - hidden_size: size of the rnn.
    - num_classes: number of classes in the dataset.
    - h_last: the hidden state vector of the core network for the last time step
    Returns
    -------
    - softmax: output probability vector over the classes.
    """
    def __init__(self, hidden_size, num_classes):
        self.w = _weight_variable((hidden_size, num_classes))
        self.b = _bias_variable((num_classes,))

    def __call__(self, h_last):
        # Take the last step only.
        logits  = tf.nn.xw_plus_b(h_last, self.w, self.b)
        pred    = tf.argmax(logits, 1)
        softmax = tf.nn.softmax(logits)

        return logits, pred, softmax

class Softattention(object):
    # h_prev: output from lstm of previous time step (shape: [batch_size, lstm_size])
    # a: Result of CNN [batch_size, conv_size * conv_size, channel_size] 
    # Attention Variables
    def __init__(self, Wa_size, Wh_size):
        self.Wa = _weight_variable((Wa_size, Wa_size))
        self.Wh = _weight_variable((Wh_size, Wa_size))
        self.Wg = _weight_variable((Wa_size, 1))
        self.b = _bias_variable((1,))
    def __call__(self, h_prev, a, i):
      
       g_list = [tf.tanh(tf.matmul(a[i], self.Wa) + tf.matmul(h_prev, self.Wh)) for i in range(len(a))]#[4 ? 256]
       m_list = [tf.nn.xw_plus_b(g_list[i], self.Wg, self.b) for i in range(len(a))]#[4 ? 1]
       m_concat = tf.concat([m_list[i] for i in range(len(a))], axis = 1)#[? 4]     
       alpha = tf.nn.softmax(m_concat)
       z_list = [tf.multiply(a[i], tf.slice(alpha, (0, i), (-1, 1))) for i in range(len(a))]
       z_stack = tf.stack(z_list, axis = 2)
       z = tf.reduce_sum(z_stack, axis = 2)
       return alpha, z,z_list,m_concat,self.Wa

class CoreNetwork(object):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.
    Concretely, it takes the images `img_ph` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.
    In other words:
        `h_t = relu( fc(h_t_prev) + fc(g_t) )`
    Args
    ----
    - batch_size: input size of the rnn.
    - loc_dim: location dim = 2
    - hidden_size: hidden size of the rnn.
    - num_glimpses: time steps of the rnn.
    - img_ph: a 4D tensor of shape (B, H, W, 1).
    Returns
    -------
    - h_ts: a 2D tensor of shape (B, hidden_size). The hidden state vector for the current timestep `t`.
    - loc_ts: a list of 2D tensor of shape (B, 2). The glimpse center sampled from guassian of all time steps.
    - mean_ts: a list of 2D tensor of shape (B, 2). The guassian mean of all time steps.
    """
    def __init__(self, batch_size, loc_dim, hidden_size, num_glimpses, rnn_batch_size, lstm_size):
        self.batch_size = batch_size
        self.loc_dim = loc_dim
        self.hidden_size = hidden_size
        self.num_glimpses = num_glimpses
        self.rnn_batch_size = rnn_batch_size
        self.lstm_size = lstm_size
        
         #cnn
        self.w_conv1 = _weight_variable((3, 3, 1, 64))
        self.b_conv1 = _bias_variable((64,))
        
    def __call__(self, img_ph, location_network1, location_network2, location_network3, location_network4, retina_sensor1, glimpse_network1,retina_sensor2, glimpse_network2, retina_sensor3, glimpse_network3, retina_sensor4, glimpse_network4, soft_attention, self_attention):
        # lstm cell
        cell = BasicLSTMCell(self.hidden_size)

        # helper func for feeding glimpses to every step of lstm
        # h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden state vector for the previous timestep `t-1`.
        loc_ts1, mean_ts1, loc_ts2, mean_ts2, loc_ts3, mean_ts3, loc_ts4, mean_ts4,loc_ts1_save,loc_ts2_save,loc_ts3_save,loc_ts4_save = [], [], [], [], [], [], [], [], [], [], [], []

        ## at time step t, location-->pths-->glimpse
        def loop_function(b_prev, _):
            # predict location from previous hidden state
            
            
            loc_t1, mean_t1 = location_network1(b_prev[0])
            loc_t2, mean_t2 = location_network2(b_prev[1])
            loc_t3, mean_t3 = location_network3(b_prev[2])
            loc_t4, mean_t4 = location_network4(b_prev[3])
            
            loc_ts1_save.append(loc_t1)
            loc_ts1.append(loc_t1)
            mean_ts1.append(mean_t1)
            
            loc_ts2_save.append(loc_t2)
            loc_ts2.append(loc_t2)
            mean_ts2.append(mean_t2)
            
            loc_ts3_save.append(loc_t3)
            loc_ts3.append(loc_t3)
            mean_ts3.append(mean_t3)
            
            loc_ts4_save.append(loc_t4)
            loc_ts4.append(loc_t4)
            mean_ts4.append(mean_t4)
            
            # crop pths from image based on the predicted location
            pths_t1 = retina_sensor1(img_ph, loc_t1)
            pths_t2 = retina_sensor2(img_ph, loc_t2)
            pths_t3 = retina_sensor3(img_ph, loc_t3)
            pths_t4 = retina_sensor4(img_ph, loc_t4)
            # generate glimpse image from current pths_t and loc_t
            glimpse_1 = glimpse_network1(pths_t1, loc_t1)#[160 256]
            glimpse_2 = glimpse_network2(pths_t2, loc_t2)
            glimpse_3 = glimpse_network3(pths_t3, loc_t3)
            glimpse_4 = glimpse_network4(pths_t4, loc_t4)
            #init_loc = tf.zeros((self.batch_size, self.loc_dim),tf.float32)
            glimpse_stack = tf.stack([glimpse_1,glimpse_2,glimpse_3,glimpse_4])#[3 ? 256]
            glimpse_unstack = tf.unstack(glimpse_stack, axis = 0)
            b = self_attention(glimpse_1, glimpse_2, glimpse_3, glimpse_4)
            
            return glimpse_unstack, b
        
        def rnn_decoder123(rnn_out, decoder_inputs, initial_state,cell, init_glimpse_1, init_glimpse_2, init_glimpse_3, init_glimpse_4, loop_function=None, scope=None):
    #with variable_scope.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs, b1_ts, b2_ts, b3_ts, b4_ts = [], [], [], [], []
            prev = None
            b_prev = None
            for i in range(self.num_glimpses+1):#inp:[160 256]就是gt
                if loop_function is not None and prev is not None:#第一次不會進
                    #with variable_scope.variable_scope("loop_function", reuse=True):
                    
                    
                    
                    inp, b = loop_function(b_prev, i)
                    #inp = [inp]
                    print('rnn1')
                #soft attention maybe added in here (without CNN) 第一次沒有loopfunction因為已在外先做了
                        #hard的lstm取代soft的lstm
                        #variable_scope.get_variable_scope().reuse_variables()
                
                if(i==0):
                    inp = decoder_inputs
                    inp  = tf.unstack(inp , axis = 0)
                    b = self_attention(init_glimpse_1, init_glimpse_2, init_glimpse_3, init_glimpse_4)
                
                        
                prev = rnn_out
                #inp = [inp]
                b_prev = b
                alpha, z,z_list,m_concat,wa = soft_attention(prev, inp, i) #alpha:[1 256] z:[1 512]
                rnn_out, state = cell(z, state)#ht:[160 256]
                outputs.append(rnn_out)
                b1_ts.append(b[0])
                b2_ts.append(b[1])
                b3_ts.append(b[2])
                b4_ts.append(b[3])
                
                print('rnn2')
                if loop_function is not None:
                    prev = rnn_out
                    print('rnn3')
                print('rnn4')
            print('rnn5')
            return outputs,state,inp,wa,alpha, b1_ts, b2_ts, b3_ts, b4_ts
        # lstm init h_t
        init_state = cell.zero_state(self.batch_size, tf.float32)

        # lstm inputs at every step
        init_loc_1 = tf.random_uniform((self.batch_size, self.loc_dim), minval=-1, maxval=1)#random at first glance
        init_loc_2 = tf.random_uniform((self.batch_size, self.loc_dim), minval=-1, maxval=1)
        init_loc_3 = tf.random_uniform((self.batch_size, self.loc_dim), minval=-1, maxval=1)
        init_loc_4 = tf.random_uniform((self.batch_size, self.loc_dim), minval=-1, maxval=1)
        #init_loc = tf.zeros((self.batch_size, self.loc_dim),tf.float32)
        
        loc_ts1_save.append(init_loc_1)
        init_pths_1 = retina_sensor1(img_ph, init_loc_1)
        init_glimpse_1 = glimpse_network1(init_pths_1, init_loc_1)#[160 256]
        
        loc_ts2_save.append(init_loc_2)
        init_pths_2 = retina_sensor2(img_ph, init_loc_2)   
        init_glimpse_2 = glimpse_network2(init_pths_2, init_loc_2)#[160 256]
        
        loc_ts3_save.append(init_loc_3)
        init_pths_3 = retina_sensor3(img_ph, init_loc_3)    
        init_glimpse_3 = glimpse_network3(init_pths_3, init_loc_3)
        
        loc_ts4_save.append(init_loc_4)
        init_pths_4 = retina_sensor4(img_ph, init_loc_4)    
        init_glimpse_4 = glimpse_network4(init_pths_4, init_loc_4)
        
        init_glimpse_stack = tf.stack([init_glimpse_1,init_glimpse_2,init_glimpse_3,init_glimpse_4])
        
        

        init_soft_rnn_out = tf.zeros([self.batch_size, self.lstm_size], tf.float32)
        
        #rnn_inputs = [init_glimpse_stack] #[  1 160 256]
        
        #init_glimpse_stack.extend([0] * self.num_glimpses)#後面加六個0

        # get hidden state of every step from lstm
        h_ts, _ ,inp,wa,alpha, b1_ts, b2_ts, b3_ts, b4_ts= rnn_decoder123(init_soft_rnn_out,init_glimpse_stack, init_state, cell, init_glimpse_1, init_glimpse_2, init_glimpse_3, init_glimpse_4, loop_function=loop_function)

        return h_ts, inp, wa, alpha, loc_ts1, mean_ts1, loc_ts2, mean_ts2, loc_ts3, mean_ts3, loc_ts4, mean_ts4, loc_ts1_save, loc_ts2_save, loc_ts3_save, loc_ts4_save, b1_ts, b2_ts, b3_ts, b4_ts



class RecurrentAttentionModel(object):
    """
    A Recurrent Model of Visual Attention (self. [1].
    self.is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    Args
    ----
    - pth_size: size of the square patches in the glimpses extracted by the retina.
    - g_size: hidden layer size of the fc layer for `phi`.
    - l_size: hidden layer size of the fc layer for `locs`.
    - glimpse_output_size: output size of glimpse network.
    - loc_dim: 2
    - std: standard deviation of the Gaussian policy.
    - hidden_size: hidden size of the rnn.
    - num_classes: number of classes in the dataset.
    - num_glimpses: number of glimpses to take per image, i.e. number of BPTT steps.

    - x: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
        state vector for the current timestep `t`.
    - mu: a 2D tensor of shape (B, 2). The mean that paself.trizes
        the Gaussian policy.
    - l_t: a 2D tensor of shape (B, 2). The location vector
        containing the glimpse coordinates [x, y] for the
        current timestep `t`.
    - b_t: a 2D vector of shape (B, 1). The baseline for the
        current time step `t`.
    - log_probas: a 2D tensor of shape (B, num_classes). The
        output log probability vector over the classes.
    """
    def __init__(self, Wa_size, Wh_size, rnn_batch_size, lstm_size, img_size, pth_size, g_size, l_size, glimpse_output_size,
                 loc_dim, std, hidden_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
                 max_gradient_norm, is_training=False):
        self.training_steps_per_epoch = training_steps_per_epoch

        with tf.variable_scope('placeholder'):
            self.img_ph = tf.placeholder(tf.float32, [None, img_size*img_size*3])
            self.lbl_ph = tf.placeholder(tf.int64, [None])
            self.is_training = tf.placeholder(tf.bool, [])

        ## init network param
        with tf.variable_scope('LocationNetwork1'):
            location_network1 = LocationNetwork1(hidden_size, loc_dim, std=std, is_sampling=self.is_training)
            
        with tf.variable_scope('LocationNetwork2'):
            location_network2 = LocationNetwork2(hidden_size, loc_dim, std=std, is_sampling=self.is_training)
            
        with tf.variable_scope('LocationNetwork3'):
            location_network3 = LocationNetwork3(hidden_size, loc_dim, std=std, is_sampling=self.is_training)
            
        with tf.variable_scope('LocationNetwork4'):
            location_network4 = LocationNetwork4(hidden_size, loc_dim, std=std, is_sampling=self.is_training)
            
        with tf.variable_scope('RetinaSensor1'):
            retina_sensor1 = RetinaSensor1(img_size, pth_size)
        
        with tf.variable_scope('RetinaSensor2'):
            retina_sensor2 = RetinaSensor2(img_size, pth_size)
            
        with tf.variable_scope('RetinaSensor3'):
            retina_sensor3 = RetinaSensor3(img_size, pth_size)
            
        with tf.variable_scope('RetinaSensor4'):
            retina_sensor4 = RetinaSensor4(img_size, pth_size)

        with tf.variable_scope('GlimpseNetwork1'):
            glimpse_network1 = GlimpseNetwork1(pth_size, loc_dim, g_size, l_size, glimpse_output_size)
            
        with tf.variable_scope('GlimpseNetwork2'):
            glimpse_network2 = GlimpseNetwork2(pth_size, loc_dim, g_size, l_size, glimpse_output_size)
            
        with tf.variable_scope('GlimpseNetwork3'):
            glimpse_network3 = GlimpseNetwork3(pth_size, loc_dim, g_size, l_size, glimpse_output_size)
            
        with tf.variable_scope('GlimpseNetwork4'):
            glimpse_network4 = GlimpseNetwork4(pth_size, loc_dim, g_size, l_size, glimpse_output_size)

        with tf.variable_scope('CoreNetwork'):
            core_network = CoreNetwork(batch_size=tf.shape(self.img_ph)[0], loc_dim=loc_dim, hidden_size=hidden_size, num_glimpses=num_glimpses, rnn_batch_size=rnn_batch_size, lstm_size=lstm_size)

        with tf.variable_scope('Baseline1'):
            baseline_network1 = BaseLineNetwork1(hidden_size)
            
        with tf.variable_scope('Baseline2'):
            baseline_network2 = BaseLineNetwork2(hidden_size)
            
        with tf.variable_scope('Baseline3'):
            baseline_network3 = BaseLineNetwork3(hidden_size)
            
        with tf.variable_scope('Baseline4'):
            baseline_network4 = BaseLineNetwork4(hidden_size)

        with tf.variable_scope('Classification'):
            classification_network = ClassificationNetwork(hidden_size, num_classes)
            
        with tf.variable_scope('Softattention'):
            soft_attention = Softattention(Wa_size, Wh_size)
            
        with tf.variable_scope('Selfattention'):
            self_attention = Selfattention()

        ## call network to build graph
        # Run the recurrent attention model for all timestep on the minibatch of images
        h_ts, self.glimpse, self.wa,self.alpha, loc_ts1,mean_ts1, loc_ts2, mean_ts2, loc_ts3, mean_ts3, loc_ts4, mean_ts4,self.loc_ts1_save,self.loc_ts2_save,self.loc_ts3_save,self.loc_ts4_save, b1_ts, b2_ts, b3_ts, b4_ts = core_network(self.img_ph, location_network1, location_network2, location_network3, location_network4, retina_sensor1, glimpse_network1, retina_sensor2, glimpse_network2, retina_sensor3, glimpse_network3,retina_sensor4, glimpse_network4, soft_attention, self_attention)

        # baselines, approximate value function based h_ts
        baselines1 = baseline_network1(b1_ts)
        baselines2 = baseline_network2(b2_ts)
        baselines3 = baseline_network3(b3_ts)
        baselines4 = baseline_network4(b4_ts)

        # make classify action at last time step
        logits, pred, self.softmax = classification_network(h_ts[-1])

        # training preparation
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(learning_rate, self.global_step, training_steps_per_epoch, learning_rate_decay_factor, staircase=True), min_learning_rate)
        self.saver = tf.train.Saver()
        #self.checkpoint = tf.train.get_checkpoint_state("C:\\Users\\WesKao\\.spyder-py3\\saver")
        ## losses
        # classification loss for classification_network, core_network, glimpse_network
        self.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))

        # RL reward for location_network
        reward = tf.stop_gradient(tf.cast(tf.equal(pred, self.lbl_ph), tf.float32))
        reward1, reward2, reward3, reward4 = alpha_reward(self.alpha, reward)
        
        rewards1 = tf.expand_dims(reward1, 1)             # [batch_sz, 1]
        rewards1 = tf.tile(rewards1, (1, num_glimpses))   # [batch_sz, timesteps]
        
        rewards2 = tf.expand_dims(reward2, 1)             # [batch_sz, 1]
        rewards2 = tf.tile(rewards2, (1, num_glimpses))   # [batch_sz, timesteps]
        
        rewards3 = tf.expand_dims(reward3, 1)             # [batch_sz, 1]
        rewards3 = tf.tile(rewards3, (1, num_glimpses))
        
        rewards4 = tf.expand_dims(reward4, 1)             # [batch_sz, 1]
        rewards4 = tf.tile(rewards4, (1, num_glimpses))
        '''
        reward*alpha
        '''
        advantages1 = rewards1 - tf.stop_gradient(baselines1) # (B, timesteps), baseline approximate func is trained by baseline loss only.
        advantages2 = rewards2 - tf.stop_gradient(baselines2)
        advantages3 = rewards3 - tf.stop_gradient(baselines3)
        advantages4 = rewards4 - tf.stop_gradient(baselines4)
        self.advantage1 = tf.reduce_mean(advantages1)
        self.advantage2 = tf.reduce_mean(advantages2)
        self.advantage3 = tf.reduce_mean(advantages3)
        self.advantage4 = tf.reduce_mean(advantages4)
        
        logll1 = _log_likelihood(mean_ts1, loc_ts1, std)  # (B, timesteps)
        logll2 = _log_likelihood(mean_ts2, loc_ts2, std)
        logll3 = _log_likelihood(mean_ts3, loc_ts3, std)
        logll4 = _log_likelihood(mean_ts4, loc_ts4, std)
        
        logllratio1 = tf.reduce_mean(logll1 * advantages1) # reduce B and timesteps
        logllratio2 = tf.reduce_mean(logll2 * advantages2)
        logllratio3 = tf.reduce_mean(logll3 * advantages3)
        logllratio4 = tf.reduce_mean(logll4 * advantages4)
        self.reward = tf.reduce_mean(reward)  # reduce batch

        # baseline loss for baseline_network, core_network, glimpse_network
        self.baselines_mse1 = tf.reduce_mean(tf.square((rewards1 - baselines1)))

        self.baselines_mse2 = tf.reduce_mean(tf.square((rewards2 - baselines2)))
        
        self.baselines_mse3 = tf.reduce_mean(tf.square((rewards3 - baselines3)))
        
        self.baselines_mse4 = tf.reduce_mean(tf.square((rewards4 - baselines4)))
        # hybrid loss
        self.loss = -logllratio1 - logllratio2 - logllratio3 - logllratio4 + self.xent + self.baselines_mse1 + self.baselines_mse2 + self.baselines_mse3 + self.baselines_mse4
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def train(self, num_steps, num_MC, BATCH_SIZE, CAPACITY, INPUT_IMAGE_LEN):
        (x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
        train_x = np.reshape(x_img_train,[50000,32*32*3])
        train_y = np.reshape(y_label_train,[50000])
        test_x = np.reshape(x_img_test,[10000,32*32*3])
        test_y = np.reshape(y_label_test,[10000])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(coord=coord)
            step_list=[]
            epoch_list=[]
            acc_list=[]
            bs_loss_list=[]
            x_loss_list=[]
            MAV_bs_loss_list=[]
            MAV_x_loss_list=[]
            MAV_acc_list=[]
            valid_loss_list=[]
            loss_list=[]
            MAV_valid_loss_list=[]
            MAV_loss_list=[]
            with open('loss.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['step', 'xent', 'baselines_mse', 'valid_loss', 'loss', 'reward'])
            with open('acc.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['step', 'acc'])
            for step in range(num_steps):
                #print("111111")
                random_idx = np.arange(train_x.shape[0])
                np.random.shuffle(random_idx)
    
                batch_index = [0,  BATCH_SIZE]

                batch_x_train = train_x[random_idx[batch_index[0]:batch_index[1]],:]
                batch_y_train = train_y[random_idx[batch_index[0]:batch_index[1]]]
                        
            # Make image as fractions for attention
                images = np.tile(batch_x_train, [num_MC, 1])
                labels = np.tile(batch_y_train, [num_MC])
                #print("222222")
                output_feed = [self.train_op, self.loss, self.xent, self.reward, self.advantage1, self.baselines_mse1, self.learning_rate,self.alpha]
                _, loss, xent, reward, advantage1, baselines_mse1, learning_rate,alpha = sess.run(output_feed, feed_dict={self.img_ph: images, self.lbl_ph: labels, self.is_training:True})
                #print("333333")
                # log
                if step and step % 100 == 0:
                    random_idx = np.arange(test_x.shape[0])
                    np.random.shuffle(random_idx)
                    batch_index = [0,  BATCH_SIZE]

                    batch_x_valid = test_x[random_idx[batch_index[0]:batch_index[1]],:]
                    batch_y_valid = test_y[random_idx[batch_index[0]:batch_index[1]]]
                    np.random.shuffle(random_idx)
                    logging.info('step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage1 = {:3.4f}\tbaselines_mse1 = {:3.4f}'.format( step, learning_rate, loss, xent, reward, advantage1, baselines_mse1))
                    v_images = np.tile(batch_x_valid, [num_MC, 1])
                    v_labels = np.tile(batch_y_valid, [num_MC])
                    valid_loss = sess.run(self.loss, feed_dict={self.img_ph: v_images, self.lbl_ph: v_labels, self.is_training:True})
                    bs_loss_list.append(baselines_mse1)
                    
                    x_loss_list.append(xent)
                    
                    valid_loss_list.append(valid_loss)
                    
                    loss_list.append(loss)
                    
                    MAV_valid_loss_list.append(np.mean(valid_loss_list[-100:]))
                    MAV_loss_list.append(np.mean(loss_list[-100:]))
                    MAV_bs_loss_list.append(np.mean(bs_loss_list[-100:]))
                    MAV_x_loss_list.append(np.mean(x_loss_list[-100:]))
                    step_list.append(step)
                    with open('loss.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        list_ = []
                        list_.append(step)
                        list_.append(xent)
                        list_.append(baselines_mse1)
                        list_.append(valid_loss)
                        list_.append(loss)
                        list_.append(reward)
                        writer.writerow(list_)
                    #print(g2)
                    #alpha_size = int(np.sqrt(alpha_.shape[1]))
                    #print(b)
                    #print(mmm)
                    '''
                    alpha_reshape = np.reshape(alpha_, (num_MC*BATCH_SIZE, num_MC*BATCH_SIZE))# 16 16
                    alpha_resize = skimage.transform.pyramid_expand(alpha_reshape, upscale = 16, sigma=20)  # 256 256
                    plt.figure()
                    plt.imshow(alpha_resize, cmap='gray')
                    '''

                # Evaluation
                if step and step % self.training_steps_per_epoch == 0:
                    correct_cnt = 0
                    TP1=TN1=FP1=FN1=TP2=TN2=FP2=FN2=0
                    
                    
                    
                    steps_per_epoch = 10000 // BATCH_SIZE
                    correct_cnt = 0
                    num_samples = steps_per_epoch * BATCH_SIZE
                    for test_step in range(steps_per_epoch):
                        random_idx = np.arange(test_x.shape[0])
                        np.random.shuffle(random_idx)
                        batch_index = [0,  BATCH_SIZE]
                        batch_x_test = test_x[random_idx[batch_index[0]:batch_index[1]],:]
                        batch_y_test = test_y[random_idx[batch_index[0]:batch_index[1]]]
                        
                        labels_bak = batch_y_test
                        # Duplicate M times
                        images = np.tile(batch_x_test, [num_MC, 1])
                        labels = np.tile(batch_y_test, [num_MC])
                        softmax = sess.run(self.softmax, feed_dict={self.img_ph: images, self.lbl_ph: labels, self.is_training:True})
                        softmax = np.reshape(softmax, [num_MC, -1, 10])
                        softmax = np.mean(softmax, 0)
                        prediction = np.argmax(softmax, 1).flatten()
                        #print(prediction)
                        #print(labels_bak)
                        correct_cnt += np.sum(prediction == labels_bak)
                        #TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2 = EVALUATION(prediction, labels_bak, 1, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, 0)
                   # PRCISION1, RECALL1, PRECISION2, RECALL2,  acc1, acc2, F_score1, F_score2 = EVALUATION(prediction, labels_bak, 1, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, 1)
                    #prnt('|ACCURACY1| = ', round(acc1,2), '|F_score1| = ', round(F_score1,2), '|PRECISION1| = ', round(PRECISION1,2), '|RECALL2| = ', round(RECALL1,2))
                   # prnt('|ACCURACY2| = ', round(acc2,2), '|F_score2| = ', round(F_score2,2), '|PRECISION2| = ', round(PRECISION2,2), '|RECALL2| = ', round(RECALL2,2))
                    plot_images_labels_prediction(images,labels,prediction,0)
                    acc = correct_cnt*1.0 / num_samples
                    logging.info('test accuracy = {}'.format(acc))
                    acc_list.append(acc)
                    MAV_acc_list.append(np.mean(acc_list[-10:]))
                    epoch_list.append(step)
                    with open('acc.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        list_ = []
                        list_.append(step)
                        list_.append(acc)
                        writer.writerow(list_)
                    
                        
                        #epoch_list2.append(step)
                        
            self.saver.save(sess,'./saver 1 glimpse stop gt lr4 csv/saved_weight.ckpt')
            print(MAV_acc_list)
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(20,10)
            Xaxis = epoch_list[len(epoch_list)-len(MAV_acc_list):]
            plt.plot(epoch_list, acc_list, color='green', linestyle='--', label = 'acc')
            plt.plot(Xaxis, MAV_acc_list, color='blue', label = 'MAVacc')
            plt.ylabel('acc')
            plt.xlabel('step')
            plt.legend(['acc','MAVacc'], loc='upper left')
            plt.savefig('./saver 1 glimpse stop gt lr4 csv/acc.jpg')
            
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(20,10)
            plt.plot(step_list,MAV_valid_loss_list, color='green', linestyle='--', label = 'valid_loss')
            plt.plot(step_list, MAV_loss_list, color='blue', label = 'loss')
            plt.ylabel('loss')
            plt.xlabel('step')
            plt.legend(['valid_loss','loss'], loc='upper left')
            plt.savefig('./saver 1 glimpse stop gt lr4 csv/loss.jpg')
            
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(20,10)
            plt.plot(step_list,MAV_x_loss_list, color='green', linestyle='--', label = 'x_loss')
            plt.ylabel('x_loss')
            plt.xlabel('step')
            plt.legend(['x_loss'], loc='upper left')
            plt.savefig('./saver 1 glimpse stop gt lr4 csv/x_loss.jpg')
            
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(20,10)
            plt.plot(step_list,MAV_bs_loss_list, color='green', linestyle='--', label = 'bs_loss')
            plt.ylabel('bs_loss')
            plt.xlabel('step')
            plt.legend(['bs_loss'], loc='upper left')
            plt.savefig('./saver 1 glimpse stop gt lr4 csv/bs_loss.jpg')
         
    def test(self, num_MC, BATCH_SIZE, CAPACITY, INPUT_IMAGE_LEN, glimpse_size):        
        (x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
        test_x = np.reshape(x_img_test,[10000,32*32*3])
        test_y = np.reshape(y_label_test,[10000])        
       #tf.reset_default_graph()
        #saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(coord=coord)

            #if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(sess, './saver 1 glimpse stop gt lr4 csv/saved_weight.ckpt')
            print("Successfully loaded:", './saver 1 glimpse stop gt lr4 csv/saved_weight.ckpt')
            print('START TESTING . . .')
            
            
            correct_cnt = 0
            steps_per_epoch = 10000 // BATCH_SIZE
            correct_cnt = 0
            num_samples = steps_per_epoch * BATCH_SIZE
            for test_step in range(steps_per_epoch):
                random_idx = np.arange(test_x.shape[0])
                np.random.shuffle(random_idx)
                batch_index = [0,  BATCH_SIZE]
                batch_x_test = test_x[random_idx[batch_index[0]:batch_index[1]],:]
                batch_y_test = test_y[random_idx[batch_index[0]:batch_index[1]]]
                
                labels_bak = batch_y_test
                # Duplicate M times
                images = np.tile(batch_x_test, [num_MC, 1])
                labels = np.tile(batch_y_test, [num_MC])
                softmax,loc_ts1_save,loc_ts2_save,loc_ts3_save,loc_ts4_save = sess.run([self.softmax,self.loc_ts1_save,self.loc_ts2_save,self.loc_ts3_save,self.loc_ts4_save], feed_dict={self.img_ph: images, self.lbl_ph: labels, self.is_training:True})
                softmax = np.reshape(softmax, [num_MC, -1, 10])
                softmax = np.mean(softmax, 0)
                prediction = np.argmax(softmax, 1).flatten()
                #print(prediction)
                #print(labels_bak)
                correct_cnt += np.sum(prediction == labels_bak)
                #TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2 = EVALUATION(prediction, labels_bak, 1, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, 0)
                # PRCISION1, RECALL1, PRECISION2, RECALL2,  acc1, acc2, F_score1, F_score2 = EVALUATION(prediction, labels_bak, 1, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, 1)
                #prnt('|ACCURACY1| = ', round(acc1,2), '|F_score1| = ', round(F_score1,2), '|PRECISION1| = ', round(PRECISION1,2), '|RECALL2| = ', round(RECALL1,2))
                # prnt('|ACCURACY2| = ', round(acc2,2), '|F_score2| = ', round(F_score2,2), '|PRECISION2| = ', round(PRECISION2,2), '|RECALL2| = ', round(RECALL2,2))
            plot_images_labels_prediction_box_printer(images,labels,prediction,loc_ts1_save,loc_ts2_save,loc_ts3_save,loc_ts4_save, glimpse_size)
            acc = correct_cnt*1.0 / num_samples
            logging.info('test accuracy = {}'.format(acc))

    def test2(self, num_MC, BATCH_SIZE, glimpse_size):        
        (x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
        test_x = np.reshape(x_img_test,[10000,32*32*3])
        test_y = np.reshape(y_label_test,[10000])     
       #tf.reset_default_graph()
        #saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(coord=coord)

            #if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(sess, './saver 1 glimpse stop gt lr4 csv/saved_weight.ckpt')
            print("Successfully loaded:", './saver 1 glimpse stop gt lr4 csv/saved_weight.ckpt')
            print('START TESTING . . .')
            
            
            correct_cnt = 0
            batch_x_test = test_x[:BATCH_SIZE]
            batch_y_test = test_y[:BATCH_SIZE]
            
            labels_bak = batch_y_test
            # Duplicate M times
            images = np.tile(batch_x_test, [num_MC, 1])
            labels = np.tile(batch_y_test, [num_MC])
            softmax,loc_ts1_save,loc_ts2_save,loc_ts3_save,loc_ts4_save = sess.run([self.softmax,self.loc_ts1_save,self.loc_ts2_save,self.loc_ts3_save,self.loc_ts4_save], feed_dict={self.img_ph: images, self.lbl_ph: labels, self.is_training:True})
            softmax = np.reshape(softmax, [num_MC, -1, 10])
            softmax = np.mean(softmax, 0)
            prediction = np.argmax(softmax, 1).flatten()
            #print(prediction)
            #print(labels_bak)
            correct_cnt += np.sum(prediction == labels_bak)
            #TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2 = EVALUATION(prediction, labels_bak, 1, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, 0)
            # PRCISION1, RECALL1, PRECISION2, RECALL2,  acc1, acc2, F_score1, F_score2 = EVALUATION(prediction, labels_bak, 1, TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2, 1)
            #prnt('|ACCURACY1| = ', round(acc1,2), '|F_score1| = ', round(F_score1,2), '|PRECISION1| = ', round(PRECISION1,2), '|RECALL2| = ', round(RECALL1,2))
            # prnt('|ACCURACY2| = ', round(acc2,2), '|F_score2| = ', round(F_score2,2), '|PRECISION2| = ', round(PRECISION2,2), '|RECALL2| = ', round(RECALL2,2))
            plot_images_labels_prediction_box_printer(images,labels,prediction,loc_ts1_save,loc_ts2_save,loc_ts3_save,loc_ts4_save, glimpse_size)
            acc = correct_cnt*1.0 / BATCH_SIZE
            logging.info('test accuracy = {}'.format(acc))

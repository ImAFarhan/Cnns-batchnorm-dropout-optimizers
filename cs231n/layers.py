from builtins import range
import numpy as np

def affine_forward(x, w, b):

    out = None

    out = np.dot(x.reshape(x.shape[0],-1),w)+b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None

    dw = np.dot(x.reshape(x.shape[0],-1).T,dout)
    dx=np.dot(dout,w.T).reshape(*x.shape)

    db=np.sum(dout,axis=0)

    return dx, dw, db


def relu_forward(x):

    out = None
    out=np.maximum(0,x)


    cache = x

    return out, cache


def relu_backward(dout, cache):

    dx, x = None, cache

    mask=x>0
    dx = np.multiply(mask,dout)

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):

    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    axis = bn_param.get("layernorm",0)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":

        batch_mean = np.mean(x,axis=0)
        batch_var = np.var(x,axis=0)
        # v-imp! we compute running stats only for testing. for train use the batch statistics rather than
        # running statistics
        if axis==0:
            running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            running_var = momentum * running_var + (1 - momentum) * batch_var
        batch_std=np.sqrt(batch_var+eps)
        std_norm = (x-batch_mean)/batch_std
        out = (gamma*std_norm)+beta
        cache = (x, gamma, beta,batch_std, std_norm,N,axis)
        

    elif mode == "test":

        std_norm = (x-running_mean)/(np.sqrt(running_var)+eps)
        out = (gamma*std_norm)+beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):

    
    x, gamma, beta,batch_std, std_norm, N, axis= cache
    dx, dgamma, dbeta = None, None, None

    dfdxh = dout*gamma
    dx=(1/(batch_std*N))*((N*dfdxh)-(np.sum(dfdxh,axis=0))-(std_norm*np.sum(dfdxh*std_norm,axis=0)))
    
    dgamma =np.sum(dout*std_norm,axis=axis) #squash after multiplying as gamma is shared between examples
    dbeta= np.sum(dout,axis = axis)
    

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):

    x, gamma, beta,batch_std, std_norm, N, axis= cache

    dx, dgamma, dbeta = None, None, None

    dfdxh = dout*gamma
    dx=(1/(batch_std*N))*((N*dfdxh)-(np.sum(dfdxh,axis=0))-(std_norm*np.sum(dfdxh*std_norm,axis=0)))
    dgamma =np.sum(dout*std_norm,axis = axis) #squash after multiplying as gamma is shared between examples
    
    dbeta= np.sum(dout,axis = axis)


    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param={}):

    out, cache = None, None
    ln_param['mode'] = 'train'
    ln_param['layernorm']= 1
    N=x.shape[0]

    out , cache = batchnorm_forward(x.T, gamma.reshape(-1,1), beta.reshape(-1,1), ln_param)
    out = out.T # as we transposed x

    return out, cache


def layernorm_backward(dout, cache):

    dout = dout.T
    dx, dgamma, dbeta = None, None, None

    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.T

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*x.shape)<p)/p
        out = x*mask
    elif mode == "test":
        out = x

    cache = mask
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    mask = cache
    dx = None

    dx =  dout * mask    
    return dx


def conv_forward_naive(x, w, b, conv_param):
  
    pad = conv_param['pad']
    stride = conv_param['stride']
    out = None

    N, C, H, W = x.shape # total images, channels, height, width
    F , _ , HH, WW = w.shape #filters , channels(same as img as extend to full depth), height, width 
    
    h_out = 1 + (H - HH + 2*pad )//stride
    w_out = 1 + (W - WW + 2*pad )//stride
    out = np.zeros((N, F, h_out, w_out))

    #padding the input
    # hi,wi = 0,0
    # from copy import deepcopy
    # for i,img in enumerate(x):
    #   img_padded =  np.pad(img, 
    #                       pad_width=((0,0),(pad,pad),(pad,pad)),
    #                       mode='constant'
    #                       , constant_values=0)
    #   k,l=0,0
    #   while hi<H:
    #     while wi<W:
    #       for j,f in enumerate(w):
    #         #print(f)
    #         local_region = img_padded[:,hi:HH+hi,wi:WW+wi].flatten()
    #         # print("filter",f)
    #         # print("local region",local_region)
    #         score = np.sum(local_region*f.flatten())+b[j]
    #         out[i,j,k,l] = score     
    #       wi += stride
    #       l+=1
    #     wi = 0
    #     hi += stride
    #     k+=1
    #     l = 0
    #   hi=0
    #   wi = 0
    #   k = 0
    # print(out)
    # print("N",N)
      
    #--------2nd implementation
    # dont forget to add bias term as you did forget it last time
    
    #lets pad all images at once
    padded_x = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad))) # 0,0 coz we dont want to pad on N,C
    ph,pw = padded_x.shape[2],padded_x.shape[3] #padded height, width

    for idx,img in enumerate(padded_x):
      img_scores = np.zeros((F,h_out,w_out)) # stores image convolution scores at each sldie
      for i_idx,i in enumerate(range(0,ph-HH+1,stride)): 
        for j_idx,j in enumerate(range(0,pw-WW+1,stride)):
          local_region=img[:,i:HH+i,j:WW+j].flatten() #flatten local region
          img_scores[:,i_idx,j_idx] = w.reshape(F,-1).dot(local_region) + b # store score at respective pos
      out[idx] = img_scores #for each image , one update 

    cache = (x, w, b, conv_param,padded_x)
    return out, cache


def conv_backward_naive(dout, cache):
    dx, dw, db = None, None, None
    x, w, b, conv_param,padded_x = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    db = dout.sum(axis=(0,2,3))
    F , C , FH, FW = w.shape #FH = filter height
    ws_flatten = w.reshape(F,-1)
    dx = np.zeros_like(x)
    h_out,w_out = dout.shape[2],dout.shape[3]
    dw = np.zeros_like(w)
    for idx,img in enumerate(x):
      dout_img = dout[idx].reshape(F,-1)
      x_temp = np.pad(np.zeros((C,H,W)),((0,0),(pad,pad),(pad,pad)))
      ph,pw = x_temp.shape[1],x_temp.shape[2]
      dx_local = dout_img.T.dot(ws_flatten)
      neuron = 0
      x_col = np.zeros((C*FH*FW,h_out*w_out))
      
      # now we have to iterate over local regions again
      for i_idx,i in enumerate(range(0,ph-FH+1,stride)): 
        for j_idx,j in enumerate(range(0,pw-FW+1,stride)):
          x_temp[:,i:FH+i,j:FW+j] += dx_local[neuron,:].reshape(C,FH,FW)
          #W
          x_col[:,neuron] = padded_x[idx,:,i:FH+i,j:FW+j].flatten()
          
          neuron += 1  
      dx[idx] = x_temp[:,pad:-pad,pad:-pad]
      
      dw += dout_img.dot(x_col.T).reshape(F,C,FH,FW)
          
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):

    out = None
    
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    h_out = 1+(H-pool_height) // stride
    w_out = 1+(W-pool_width) // stride

    out = np.zeros((N,C,h_out,w_out))
    # pooling is applied on each activation map independently
    for idx, act_map in enumerate(x):
      for i_idx,i in enumerate(range(0,H-pool_height+1,stride)):
        for j_idx,j in enumerate(range(0,W-pool_width+1,stride)):
          scores = act_map[:,i:pool_height+i,j:pool_width+j].reshape(C,-1).max(axis=1)
          out[idx,:,i_idx,j_idx]=scores
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):

    dx = None

    x, pool_param = cache

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    h_out = 1+(H-pool_height) // stride
    w_out = 1+(W-pool_width) // stride
    dx = np.zeros_like(x)

    for idx, act_map in enumerate(x):
      dout_row = dout[idx].reshape(C,-1)
      neuron = 0
      x_temp = np.zeros_like(x) 
      for i_idx,i in enumerate(range(0,H-pool_height+1,stride)):
        for j_idx,j in enumerate(range(0,W-pool_width+1,stride)):
          pool_values = act_map[:,i:pool_height+i,j:pool_width+j].reshape(C,-1)
          max_val_idx = pool_values.argmax(axis=1)
          local_region= np.zeros_like(pool_values)
          local_region[np.arange(C),max_val_idx]= dout_row[:,neuron]
          dx[idx,:,i:pool_height+i,j:pool_width+j]=local_region.reshape(C,pool_height,pool_width)
          neuron += 1 

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):

    out, cache = None, None


    N, C, H , W = x.shape
    data = x.reshape(-1,C)
    out, cache = batchnorm_forward(data, gamma, beta, bn_param)
    out = out.reshape(N,C,H,W)

    return out, cache


def spatial_batchnorm_backward(dout, cache):

    dx, dgamma, dbeta = None, None, None


    N, C, H , W = dout.shape
    dout_channel = dout.reshape(-1,C)
    dx, dgamma , dbeta = batchnorm_backward(dout_channel, cache)
    dx= dx.reshape(N,C,H, W)


    return dx, dgamma, dbeta

def svm_loss(x, y):

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    # we already took the exponent on z
    # log(exp(shifted_logits))/log(z)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

    
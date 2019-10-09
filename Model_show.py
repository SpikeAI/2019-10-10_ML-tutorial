import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def transform_img(i) :
    
    if len(i.shape)==4: i=i[0]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    i = i.detach().numpy().transpose((1, 2, 0))
    i = std * i + mean
    i = np.clip(i, 0, 1)
    
    return i


def featuremap(i, model, conv, num_trial, num_featuremap) :
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    conv.register_forward_hook(get_activation('conv'))
    data = i
    data.unsqueeze_(0)
    output = model(data)

    act = activation['conv'].squeeze()
    
    return act[num_featuremap]

def kernel(conv, num_kernel, num_kernel_previous_layer) :
    return conv.weight[num_kernel][num_kernel_previous_layer]

def img_conv(fig, i, model, conv, titre, num_trial, num_featuremap, num_featuremap_previous_layer, conv2=False, i_0=None) :
    
    if conv2 : i0 = i_0
    else :     i0 = i
    
    w = kernel(conv, num_featuremap, num_featuremap_previous_layer)
    w = w.view(1, w.shape[-2], w.shape[-1])
    
    i1 = featuremap(i, model, conv, num_trial, num_featuremap)
    i1 = i1.view(1, i1.shape[-2], i1.shape[-1])
    
    
    i2 = F.relu(i1)
    i2 = i2.view(1, i2.shape[-2], i2.shape[-1])
    
    
    i3 = F.max_pool2d(i2, 2, 2)

    e1 = 0.1
    e = 0.07
    a = (i0.shape[-1]/100)/1
    b = (w.shape[-1]/100)/1
    c = (i1.shape[-1]/100)/1
    d = (c/2)/1

    ax0 = fig.add_axes([e1, e1, a, a])
    ax1 = fig.add_axes([e1+a+e, e1+(a/2)-(b/2), b, b])
    ax2 = fig.add_axes([e1+a+e+b+e, e1+(a/2)-(c/2), c, c])
    ax3 = fig.add_axes([e1+a+e+b+e+c+e, e1+(a/2)-(c/2), c, c])
    ax4 = fig.add_axes([e1+a+e+b+e+c+e+c+e, e1+(a/2)-(d/2), d, d])
    
    
    arg = dict(fontsize=40, ha='center', va='center', transform=fig.transFigure)
    plt.text(e1+a+(e/2)-0.01, e1+(a/2), r'$\otimes$', **arg)
    plt.text(e1+a+e+b+(e/2)-0.01, e1+(a/2), r'=', **arg)
    plt.text(e1+a+e+b+e+c+(e/2)-0.01, e1+(a/2), r'$\to$', **arg)
    plt.text(e1+a+e+b+e+c+e+c+(e/2)-0.01, e1+(a/2), r'$\to$', **arg)

    arg1 = dict(fontsize=25)#, ha='center', va='center')
    
    if conv2 :
    
        ax5 = fig.add_axes([e1+a+e+b+e+c+e+c+e+d+e+0.1, e1, 0.01, a])
        plt.text(e1+a+e+b+e+c+e+c+e+d+e, e1+(a/2), r'$\to$', **arg)
        flatening = transform_img(i3)#.view(-1, i3.shape[0]*i3.shape[1])
        flatening = np.reshape(flatening, (4*4, 1, 3))
        
        ax5.set_title('Flattening x1')
        ax5.imshow(flatening)
        ax5.set_xticks([]) ; ax5.set_yticks([])
        ax5.set_xlabel('1', **arg1) ; ax5.set_ylabel('4x4x50', **arg1)

    for a, inp, title in zip([ax0, ax1, ax2, ax3, ax4], [i0, w, i1, i2, i3], titre) :

        a.set_title(title)
        a.imshow(transform_img(inp))
        a.set_xticks([]) ; a.set_yticks([])
        a.set_xlabel(inp.shape[-2], **arg1) ; a.set_ylabel(inp.shape[-1], **arg1)
    
    return i3

def Plot_model(dataset, model, num_trial, num_featuremap1, num_featuremap2) :
    
    fig = plt.figure(figsize=(15,15))
    titre = ['original image','kernel x20','feature map x20','after ReLu x20','after MaxPooling x20']
    i = img_conv(fig, dataset[num_trial][0], model, model.conv1, titre, num_trial, num_featuremap1, 0)

    fig.suptitle('first convolution', y=0.5, fontsize=55)
    plt.show()
    
    fig = plt.figure(figsize=(15,15))
    titre = ['image after first convolution x20','kernel x50','feature map x50',
             'after ReLu x50','after MaxPooling x50']
    i = img_conv(fig, dataset[num_trial][0], model, model.conv2, titre, num_trial, num_featuremap2, num_featuremap1, conv2=True, i_0=i)
    fig.suptitle('second convolution', y=0.3, x=0.37, fontsize=45)
    plt.show()
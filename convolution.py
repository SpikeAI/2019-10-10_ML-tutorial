import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def transform_img(i) :
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    i = i.detach().numpy().transpose((1, 2, 0))
    i = std * i + mean
    i = np.clip(i, 0, 1)
    
    return i

def img_conv(fig, i, conv, titre, conv2=False) :
    
    w = conv.weight[0]
    
    i1 = conv(i.view(1, 1, i.shape[2], i.shape[2]))
    i1 = i1[0][0].view(1, i1.shape[2], i1.shape[2])
    
    i2 = F.relu(i1)
    
    i3 = F.max_pool2d(i2, 2, 2)

    e1 = 0.1
    e = 0.07
    a = (i.shape[1]/100)/1
    b = (w.shape[1]/100)/1
    c = (i1.shape[1]/100)/1
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

    for a, inp, title in zip([ax0, ax1, ax2, ax3, ax4], [i, w, i1, i2, i3], titre) :

        a.set_title(title)
        a.imshow(transform_img(inp))
        a.set_xticks([]) ; a.set_yticks([])
        
        a.set_xlabel(inp.shape[1], **arg1) ; a.set_ylabel(inp.shape[2], **arg1)
    
    fig.tight_layout()
    return i3

def Plot_img_conv(inputs) :
    fig = plt.figure(figsize=(15,15))
    conv1 = nn.Conv2d(1, 20, 5, 1)
    i = img_conv(fig, inputs, conv1, ['image original', 'kernel x20',
                                              'carte de caractéristique x20',
                                              'après ReLu x20', 'après MaxPooling x20'])

    fig.suptitle('première convolution', y=0.5, fontsize=55)
    plt.show()
    
    fig = plt.figure(figsize=(15,15))
    conv2 = nn.Conv2d(1, 50, 5, 1)
    i = img_conv(fig, i, conv2, ['image après première convolution x20', 'kernel x50',
                                 'carte de caractéristique x50',
                                 'après ReLu x50', 'après MaxPooling x50'], conv2=True)
    fig.suptitle('deuxième convolution', y=0.3, x=0.37, fontsize=45)
    plt.show()
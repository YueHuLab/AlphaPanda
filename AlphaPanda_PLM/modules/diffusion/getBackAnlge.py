import torch
from torch import tensor
import copy
import glob
import pickle


gly_CB_mu = torch.tensor([-0.5311191 , -0.75842446,  1.2198311 ]) #pickle.load(open("pkl/CB_mu.pkl", "rb"))


def get_len(v):
    #return tensor.sqrt(tensor.sum(v ** 2, -1))
    #return tensor.sum(v ** 2, -1).sqrt() #huyue
    v=v.type(torch.DoubleTensor)
    return v.norm(2,-1).type(torch.FloatTensor) #huyue


def get_unit_normal(ab, bc):
    ab=ab.type(torch.FloatTensor)
    bc=bc.type(torch.FloatTensor)
    n = torch.cross(ab, bc, -1)
    length = get_len(n)
    if len(n.shape) > 2:
        length = length[..., None]
    length=torch.where(length!=0,length,1)
    return n / length
    #if length.all()!=0 : #huyue
    #    return n / length
    #else:
         #return n / tensor.clip(length,0.1,0.2)
    #    return n



def get_angle(v1, v2):
    # get in plane angle between v1, v2 -- cos^-1(v1.v2 / ||v1|| ||v2||)
    #return tensor.arccos(tensor.sum(v1 * v2, -1) / get_len(v1) * get_len(v2)) # if add kuohao ??????? huyue
    length1=get_len(v1.type(torch.DoubleTensor))
    length2=get_len(v2.type(torch.DoubleTensor))
    length1=torch.where(length1!=0,length1,1)
    length2=torch.where(length2!=0,length2,1)

    return torch.arccos(torch.clip(torch.sum(v1 * v2, -1) / (length1 * length2),-1,1).type(torch.DoubleTensor)).type(torch.FloatTensor)

    #if get_len(v1).all()!=0 and get_len(v2).all()!=0:
    #    return torch.arccos(torch.clip(torch.sum(v1 * v2, -1) / get_len(v1.type(torch.DoubleTensor)) * get_len(v2.type(torch.DoubleTensor)),-1,1).type(torch.DoubleTensor)).type(torch.FloatTensor)
    #else:
    #    return torch.arccos(torch.clip(torch.sum(v1 * v2, -1),-1,1).type(torch.DoubleTensor)).type(torch.FloatTensor)
        #return tensor.arccos(0) #huyue



def bdot(a, b):
    a=a.type(torch.DoubleTensor)
    b=b.type(torch.DoubleTensor)
    return torch.matmul(a, b).type(torch.FloatTensor)


def return_align_f(axis, theta):
    c_theta = torch.cos(theta)[..., None]
    s_theta = torch.sin(theta)[..., None]
    #print("f_rot C theta shape\n")
    #print(c_theta.shape)
    #print("f_rot shape\n")
    #f_rot = lambda v: c_theta * v + s_theta * tensor.cross(axis, v, axis=-1) + (1 - c_theta) * bdot(axis, v.transpose(0, 2, 1)) * axis
    ab=axis.type(torch.FloatTensor)
    #bc=v.type(torch.FloatTensor)
    f_rot = lambda v: c_theta * v + s_theta * torch.cross(ab, torch.FloatTensor(v),-1) + (1 - c_theta) * bdot(axis, v.transpose( 2, 1)) * axis
    
    return f_rot


def return_batch_align_f(axis, theta, n):
    # n is total number of atoms
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    #axis = tensor.repeat(axis, n, axis=1)[:, :, None]
    #c_theta = tensor.repeat(c_theta, n, axis=1)[:, :, None, None]
    #s_theta = tensor.repeat(s_theta, n, axis=1)[:, :, None, None]
    #print("siez axis shape before\n")
    #print(axis.shape)
    #print("siez axis shape\n")
    #print("siez c_theta shape beofre\n")
    #print(c_theta.shape)
    #print("siez axis shape\n")
    
    axis = axis.repeat(1,n,1,1)
    #print("siez axis shapei middle\n")
    #print(axis.shape)
    #print("siez axis shape\n")
    axis = axis[:, :, None]
    c_theta = c_theta.repeat(1, n,1)[:, :, None, None]
    s_theta = s_theta.repeat(1, n,1)[:, :, None, None]
    #print("siez axis shape\n")
    #print(axis.shape)
   # print("siez axis shape\n")
   # print("siez c_theta shape\n")
   # print(c_theta.shape#)
    #print("siez axis shape\n")

    #axis = axis.repeat(1,1,n,1)[:, :, None]
    #c_theta = c_theta.repeat(1, 1,n,1)[:, :, None, None]
    #s_theta = s_theta.repeat(1, 1,n,1)[:, :, None, None]

    #f_rot = lambda v: c_theta * v + s_theta * tensor.cross(axis, v, axis=-1) + (1 - c_theta) * bdot(axis, v.transpose(0, 1, 3, 2)) * axis
    f_rot = lambda v: c_theta * v + s_theta * torch.cross(axis, v, -1) + (1 - c_theta) * bdot(axis, v.transpose(3, 2)) * axis
    return f_rot



def get_batch_N_CA_C_align_back(normal, r, n): #huyue
    # get fn to align n to positive z_hat, via rotation about x axis (assume N-CA already along x_hat)
    # r is number of residues
    #z = tensor.repeat(tensor.array([[0, 0, 1]]), r, 0)[:, None]
    z = torch.tensor([[0, 0, 1]]).repeat(r,1)[:, None]
    #z = torch.tensor([[0, 0, 1]]).repeat(1,1,r)[:, None]
    #print("siez Z shape\n")
    #print(z.shape)
    #print("siez Z shape\n")
    theta = get_angle(z,normal)
    axis = get_unit_normal(z,normal)
    #print("siez Z axix batch N CA shape\n")
    #print(axis.shape)
    #print("siez Z shape\n")
    #print("siez Z theta batch N CA shape\n")
    #print(theta.shape)
    #print("siez Z shape\n")
    return return_align_f(axis, theta), return_batch_align_f(axis, theta, n=n)



def get_batch_N_CA_align_back(v, r, n):  #huyue
    # assuming ca is at (0,0,0), return fn to batch align CA--N to positive x axis
    # v = n - ca
    #x = tensor.repeat(tensor.array([[1, 0, 0]])[None], r, 0)
    x = torch.tensor([[1, 0, 0]])[None].repeat(r,1,1)
    #x = torch.tensor([[1, 0, 0]])[None].repeat(1,1,r)
    #print("siez X shape\n")
    #print(x.shape)
    #print("siez X shape\n")
    axis = get_unit_normal(x,v)
    theta = get_angle(x,v)
    #print("siez X axix batch N CA shape\n")
    #print(axis.shape)
    #print("siez X shape\n")
    #print("siez X theta batch N CA shape\n")
    #print(theta.shape)
   # print("siez X shape\n")
    return return_align_f(axis, theta), return_batch_align_f(axis, theta, n=n)



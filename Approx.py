#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml
import pprint
from Dimensionner import Dimensionner
import matplotlib.pyplot as plt


def load_config(file_name):
    if file_name:
        with open(file_name, 'r') as file:
            config = yaml.safe_load(file)
    return config


# In[2]:


config = load_config("init_config.yml")


# In[3]:


pprint.pprint(config)


# In[4]:


spectral_samples = 10
config["spectral_samples"] = spectral_samples


# In[5]:


dimensionner = Dimensionner(config)


# In[6]:


list_X_dmd, list_Y_dmd, list_wavelengths = dimensionner.propagate()


# In[7]:


X_cam = dimensionner.X_cam
Y_cam = dimensionner.Y_cam


# In[8]:




# In[9]:


from utils.functions_retropropagating import *


# In[10]:


import jax.numpy as np
from jax import grad, jacfwd






# In[11]:


F = dimensionner.config["system architecture"]["focal lens 1"]
A = np.radians(dimensionner.config["system architecture"]["dispersive element 1"]["A"])
alpha_c = dimensionner.alpha_c
delta_alpha_c = np.radians(dimensionner.config["system architecture"]["dispersive element 1"]["delta alpha c"])
delta_beta_c = np.radians(dimensionner.config["system architecture"]["dispersive element 1"]["delta beta c"])


# In[12]:


propagate_through_arm_scalar(X_cam[0,0],Y_cam[0,0],1.518,A,F,alpha_c, delta_alpha_c,delta_beta_c)


# In[ ]:





# In[28]:


jacobian_propagate = jacfwd(dimensionner.propagate_through_arm_scalar_no_params)

X_cam_float = np.float32(X_cam[0,0])
Y_cam_float = np.float32(Y_cam[0,0])
A_float = np.float32(A)
F_float = np.float32(F)
alpha_c_float = np.float32(alpha_c)
delta_alpha_c_float = np.float32(delta_alpha_c)
delta_beta_c_float = np.float32(delta_beta_c)

jacobian_at_x0 = np.array(jacobian_propagate((X_cam_float,Y_cam_float,1.518)))[:,:,0]

print(jacobian_at_x0)


# Use the Jacobian matrix to approximate the function's values at a new point
X_cam_new = X_cam_float + 50
Y_cam_new = Y_cam_float - 50
n_new = 1.518 -0.001

x0 = np.array([X_cam_float,Y_cam_float,1.518])
x_new = np.array([X_cam_new,Y_cam_new,n_new])

f_x0 = np.array(dimensionner.propagate_through_arm_scalar_no_params(x0))
delta_x = x_new - x0

print(jacobian_at_x0.shape)
print(np.dot(jacobian_at_x0, delta_x).shape)


approx_f_x_new = f_x0[:,0] + np.dot(jacobian_at_x0, delta_x)


print(approx_f_x_new)

f_x0_delta = np.array(dimensionner.propagate_through_arm_scalar_no_params(x_new))
print(f_x0_delta)


# In[30]:
import time
t_0 = time.time()

jacobian_propagate2 = jacfwd(dimensionner.propagate_through_arm_scalar_no_params)
t_1 = time.time()
print("Time to compute jacobian: ", t_1 - t_0,"s")

X_cam_float = np.float32(X_cam).flatten()
Y_cam_float = np.float32(Y_cam).flatten()
n_array_float = np.full(X_cam.shape,sellmeier(550)).flatten()

num_wavelengths = 1000

import numpy as np
wavelengths = np.linspace(dimensionner.config["spectral range"]["wavelength min"],
                          dimensionner.config["spectral range"]["wavelength max"],
                          num_wavelengths)
# Calculate the refractive index for each wavelength
n_array_float_repeated = np.array([sellmeier(wavelength) for wavelength in wavelengths])
# Repeat for each X_cam and Y_cam position
n_array_float = np.repeat(n_array_float_repeated, len(X_cam_float))
X_cam_float = np.repeat(X_cam_float, num_wavelengths)
Y_cam_float = np.repeat(Y_cam_float, num_wavelengths)
# List of wavelengths to use


A_float = np.float32(A)
F_float = np.float32(F)
alpha_c_float = np.float32(alpha_c)
delta_alpha_c_float = np.float32(delta_alpha_c)
delta_beta_c_float = np.float32(delta_beta_c)

x0 = np.array([X_cam_float,Y_cam_float,n_array_float])

from jax import vmap

import time
t_0 = time.time()
jacobian_at_each_point = vmap(jacobian_propagate2,in_axes=1)(x0)



jacobian_at_each_point = np_jax.array([jacobian_at_each_point[0][:,0,:],jacobian_at_each_point[1][:,0,:]])
jacobian_at_each_point = np.transpose(jacobian_at_each_point, (0, 2, 1))
t_1 = time.time()
print("Time to compute jacobian at each point: ", t_1 - t_0,"s")
# jacobian_at_x0 = np.array(jacobian_propagate2(X_cam_float,Y_cam_float,n_array_float))[:,:,0]


# In[31]:




# In[36]:
import numpy
import random

# Use the Jacobian matrix to approximate the function's values at a new point
X_cam_new = X_cam_float + numpy.random.normal(0, 100, X_cam_float.shape)
Y_cam_new = Y_cam_float - numpy.random.normal(0, 100, X_cam_float.shape)
n_new = n_array_float - numpy.random.normal(0, 0.001, X_cam_float.shape)

x0 = np.array([X_cam_float,Y_cam_float,n_array_float])
x_new = np.array([X_cam_new,Y_cam_new,n_new])

f_x0 = np.array(dimensionner.propagate_through_arm_vector_no_params((x0[0],x0[1],x0[2])))
delta_x = x_new - x0


print(jacobian_at_each_point.shape,delta_x.shape)
# Add an extra dimension to delta_x
delta_x = np.expand_dims(delta_x, axis=1)  # shape is now (3, 1, 2035)

print(f_x0.shape)
print(np.einsum('ijk,jik->ik', jacobian_at_each_point, delta_x))
print(np.einsum('ijk,jik->ik', jacobian_at_each_point, delta_x).shape)
# Perform the dot product


t_0 = time.time()
approx_f_x_new = f_x0+ np.einsum('ijk,jik->ik', jacobian_at_each_point, delta_x)
t_1 = time.time()
print("Time to compute approx_f_x_new: ", t_1 - t_0,"s")


f_x0_delta = np.array(dimensionner.propagate_through_arm_vector_no_params(x_new))


plt.scatter(X_cam_float, Y_cam_float)
plt.scatter(X_cam_new, Y_cam_new)
plt.show()


plt.scatter(approx_f_x_new[0,:].reshape((X_cam.shape[0],X_cam.shape[1],num_wavelengths))[:,:,0],approx_f_x_new[1,:].reshape((X_cam.shape[0],X_cam.shape[1],num_wavelengths))[:,:,0])
plt.scatter(f_x0_delta[0,:].reshape((X_cam.shape[0],X_cam.shape[1],num_wavelengths))[:,:,0],f_x0_delta[1,:].reshape((X_cam.shape[0],X_cam.shape[1],num_wavelengths))[:,:,0])
plt.show()

import random
import torch
from training import train
from functions import *
import os 



def get_arguments():
    args = DotDict()
    args.not_cuda = False
    args.netG = ''  #Path to a pre-trained generator network, if any. Default is an empty string.
    args.netD = ''  #Path to a pre-trained discriminator network, if any. Default is an empty string.
    args.manualSeed = None   #Seed for random number generation.
    args.nc_z = 3    #Number of channels in the noise input. Default is 3.
    args.nc_im = 3  #Number of channels in the input image. Default is 3.
    args.out = 'Output'   #Directory to save the output results.
    args.nfc = 32      #Number of feature channels in the first layer of the generator. Default is 32.
    args.min_nfc = 32   #Minimum number of feature channels in the generator. Default is 32.
    args.ker_size = 3  #Kernel size for the convolutional layers. Default is 3.
    args.num_layer = 5   #Number of layers in the generator and discriminator networks. Default is 5.
    args.stride = 1     #Stride for the convolutional layers. Default is 1.
    args.padd_size = 0    #Padding size for the convolutional layers. Default is 0.
    args.scale_factor = 0.75   #Scale factor for creating image pyramids. Default is 0.75.
    args.noise_amp = 0.1   #Amplitude of the noise added at each scale. Default is 0.1.
    args.min_size = 25   #Minimum size of the images in the pyramid. Default is 25.
    args.max_size = 250   #Maximum size of the images in the pyramid. Default is 250
    args.niter = 1000  #Number of iterations for training at each scale. Default is 10.
    args.gamma = 0.1   #Factor for learning rate decay. Default is 0.1.
    args.lr_g = 0.0005    #Learning rate for the generator. Default is 0.0005.
    args.lr_d = 0.0005    #Learning rate for the discriminator. Default is 0.0005.
    args.beta1 = 0.5    #Beta1 parameter for Adam optimizer. Default is 0.5.
    args.Gsteps = 3    #Number of steps to update the generator in each iteration. Default is 3.
    args.Dsteps = 3    #Number of steps to update the discriminator in each iteration. Default is 3.
    args.lambda_grad = 0.1    #Gradient penalty coefficient for WGAN-GP. Default is 0.1.
    args.alpha = 10     # Weight for the reconstruction loss. Default is 10
    args.input_dir = 'Input'  #Directory containing the input images.
    args.input_name = ''   #Name of the input image file.
    args.mode = 'train' #Mode of operation

    return args





# Step 1: Get the command-line arguments or configuration settings
args = get_arguments()
opt = args

# Step 2: Set the name of the input image
opt.input_name = 'etretat.jpg'  # You can change this to the desired input image name

# Step 3: Finalize the configuration settings
opt = post_config(opt)

# Step 4: Initialize lists to store models, latent vectors, real images, and noise amplification factors
Gs = []         # List to store the generator networks for each scale
Zs = []         # List to store the latent vectors for each scale
reals = []      # List to store the real images for each scale
NoiseAmp = []   # List to store the noise amplification factors for each scale

# Step 5: Generate the directory path to save the trained models
dir2save = generate_dir2save(opt)

# Step 6: Check if the directory already exists
if os.path.exists(dir2save):
    print('Directory for Trained model already exists')  # If it exists, print a message
else:
    # If it does not exist, create the directory
    try:
        os.makedirs(dir2save)
    except OSError:
        pass  # If there's an error in creating the directory, just pass

# Step 7: Read and preprocess the input image
real = read_image(opt)

# Step 8: Adjust the scales of the image according to the configuration settings
adjust_scales2image(real, opt)


# Step 9: Train the model using the configurations and initialized lists
train(opt, Gs, Zs, reals, NoiseAmp)


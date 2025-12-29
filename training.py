import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import init_models, InceptionV3
from functions import *
from imresize import imresize

def update_discriminator(netD, real, fake, optimizerD, opt):
    netD.zero_grad()

    # Forward pass real images through discriminator
    output_real = netD(real).to(opt.device)
    errD_real = -output_real.mean()  # Calculate the real image loss
    errD_real.backward()  # Backpropagate the real image loss
    D_x = -errD_real.item()  # Store D(x) for real images

    # Forward pass fake images through discriminator
    output_fake = netD(fake.detach())
    errD_fake = output_fake.mean()  # Calculate the fake image loss
    errD_fake.backward()  # Backpropagate the fake image loss
    D_G_z = output_fake.mean().item()  # Store D(G(z)) for fake images

    gradient_penalty = calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
    gradient_penalty.backward()  # Backpropagate the gradient penalty

    errD = errD_real + errD_fake + gradient_penalty  # Combine losses
    optimizerD.step()  # Update discriminator parameters

    return errD, D_x, D_G_z

def update_generator(netG, netD, real, fake, optimizerG, z_opt, z_prev, alpha, opt):
    netG.zero_grad()

    # Forward pass through discriminator
    output = netD(fake)
    errG = -output.mean()  # Calculate the generator loss
    errG.backward()  # Backpropagate the generator loss

    rec_loss = 0
    if alpha != 0:
        loss = nn.MSELoss()
        Z_opt = opt.noise_amp * z_opt+ z_prev
        rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev.detach()), real)
        rec_loss.backward()  # Backpropagate the reconstruction loss
        rec_loss = rec_loss.detach()  # Detach to avoid inplace operation issues

    optimizerG.step()  # Update generator parameters

    return errG, rec_loss



def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt):
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging
    real = reals[len(Gs)].float()  # Get the current real image at the current scale
    opt.nzx = real.shape[2]  # Set the size of the noise along x-dimension
    opt.nzy = real.shape[3]  # Set the size of the noise along y-dimension
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride  # Calculate receptive field size
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)  # Calculate padding for noise
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)  # Calculate padding for image
    m_noise = nn.ZeroPad2d(int(pad_noise))  # Create padding layer for noise
    m_image = nn.ZeroPad2d(int(pad_image))  # Create padding layer for image

    alpha = opt.alpha  # Set alpha parameter

    fixed_noise = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device).float()  # Generate fixed noise
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device).float()  # Initialize z_opt with zeros
    z_opt = m_noise(z_opt)  # Apply noise padding

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))  # Create optimizer for the discriminator
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))  # Create optimizer for the generator
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)  # Learning rate scheduler for discriminator
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)  # Learning rate scheduler for generator

    # Initialize Inception model for SIFID
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[64]
    model = InceptionV3([block_idx])
    model = model.to(opt.device)

    sifid2plot = []  # List to store SIFID scores

    for epoch in range(opt.niter):
        if not Gs:  # If there are no previous scales
            z_opt = generate_noise([1, opt.nzx, opt.nzy], device=opt.device).float()  # Generate new noise
            z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))  # Expand and pad the noise
            noise_ = generate_noise([1, opt.nzx, opt.nzy], device=opt.device).float()  # Generate another noise
            noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))  # Expand and pad the noise
        else:
            noise_ = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device).float()  # Generate new noise for the current scale
            noise_ = m_noise(noise_)  # Pad the noise

        if epoch == 0:  # At the start of training
            if not Gs:  # If there are no previous scales
                prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device).float()  # Initialize prev with zeros
                in_s = prev  # Set in_s to prev
                prev = m_image(prev)  # Apply image padding
                z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device).float()  # Initialize z_prev with zeros
                z_prev = m_noise(z_prev)  # Apply noise padding
                opt.noise_amp = 1  # Set noise amplitude
            else:  # If there are previous scales
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt).float()  # Draw concatenated image with random noise
                prev = m_image(prev)  # Apply image padding
                z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt).float()  # Draw concatenated image with reconstruction noise
                criterion = nn.MSELoss()  # Define Mean Squared Error loss
                RMSE = torch.sqrt(criterion(real, z_prev))  # Calculate Root Mean Squared Error between real and z_prev
                opt.noise_amp = opt.noise_amp_init * RMSE  # Adjust noise amplitude based on RMSE
                z_prev = m_image(z_prev)  # Apply image padding
        else:
            prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt).float()  # Draw concatenated image with random noise
            prev = m_image(prev)  # Apply image padding

        if not Gs:
            noise = noise_
        else:
            noise = opt.noise_amp * noise_ + prev  # Combine noise and previous image

        fake = netG(noise.detach(), prev.float())  # Generate fake image using the generator

        # Update discriminator
        for j in range(opt.Dsteps):
            errD, D_x, D_G_z = update_discriminator(netD, real, fake, optimizerD, opt)  # Update discriminator

        # Update generator
        for j in range(opt.Gsteps):
            fake = netG(noise.detach(), prev.float())  # Generate fake image using the generator
            errG, rec_loss = update_generator(netG, netD, real, fake, optimizerG, z_opt, z_prev, alpha, opt)  # Update generator

        fake_generated = netG(z_opt.detach(), z_prev.float())  # Generate fake image using the generator

        # Calculate SIFID score
        mu1, sigma1 = calculate_activation_statistics(real.unsqueeze(0), model, dims=64, cuda=False)
        mu2, sigma2 = calculate_activation_statistics(fake_generated.detach().unsqueeze(0), model, dims=64, cuda=False)
        sifid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        sifid2plot.append(sifid_score)

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))  # Print progress

        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            plt.imsave('%s/fake_sample.png' % (opt.outf), convert_image_np(fake.detach()), vmin=0, vmax=1)  # Save fake image
            plt.imsave('%s/G(z_opt).png' % (opt.outf), convert_image_np(fake_generated.detach()), vmin=0, vmax=1)  # Save generator output for z_opt
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))  # Save z_opt

    # Display the real and fake images in the same row at the end of training each scale
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(convert_image_np(real), vmin=0, vmax=1)
    ax[0].set_title('Real Image')
    ax[0].axis('off')

    ax[1].imshow(convert_image_np(fake_generated.detach()), vmin=0, vmax=1)
    ax[1].set_title('Generated fake Image')
    ax[1].axis('off')

    plt.show()

    # Plot SIFID scores
    plt.figure(figsize=(10, 5))
    plt.plot(sifid2plot, label='SIFID Score')
    plt.xlabel('Epoch')
    plt.ylabel('SIFID')
    plt.title('SIFID Score per Epoch')
    plt.legend()
    plt.show()

    save_networks(netG, netD, z_opt, opt)  # Save the networks
    return z_opt, in_s, netG  # Return the optimal z, input scale, and generator


def train(opt, Gs, Zs, reals, NoiseAmp):
    # Read and preprocess the input image
    real_ = read_image(opt)
    in_s = 0  # Initialize the starting image
    scale_num = 0  # Initialize the scale level
    real = imresize(real_, opt.scale1, opt)  # Resize the input image to the first scale ## The input image will have max dim = 250 and will be resized accordingly
    reals = creat_reals_pyramid(real, reals, opt)  # Create a pyramid of resized images
    nfc_prev = 0  # Initialize the previous number of feature channels

    while scale_num < opt.stop_scale + 1:
        # Update the number of feature channels based on the scale level
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        # Generate directory to save the models
        opt.out_ = generate_dir2save(opt)
        opt.outf = f'{opt.out_}/{scale_num}'
        os.makedirs(opt.outf, exist_ok=True)  # Create the directory if it doesn't exist

        # Save the real image at the current scale
        plt.imsave(f'{opt.outf}/real_scale.png', convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        # Initialize the models for the current scale
        D_curr, G_curr = init_models(opt)
        if nfc_prev == opt.nfc:
            # Load pre-trained weights if the number of feature channels hasn't changed
            G_curr.load_state_dict(torch.load(f'{opt.out_}/{scale_num-1}/netG.pth'))
            D_curr.load_state_dict(torch.load(f'{opt.out_}/{scale_num-1}/netD.pth'))

        # Train the models for the current scale
        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)

        # Set the models to evaluation mode and reset their gradients
        G_curr = reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = reset_grads(D_curr, False)
        D_curr.eval()

        # Append the trained models and parameters to their respective lists
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        # Save the current state of the training process
        torch.save(Zs, f'{opt.out_}/Zs.pth')
        torch.save(Gs, f'{opt.out_}/Gs.pth')
        torch.save(reals, f'{opt.out_}/reals.pth')
        torch.save(NoiseAmp, f'{opt.out_}/NoiseAmp.pth')

        # Move to the next scale
        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr  # Free up memory

    return

# VAE_Mnist_Exploration
VAE latent space exploration for generating mnist digits

This is a very simple app for exploring latent space of a vae model that generates mnist digits. The vae code for building the model can be found in the Autoencoder repository, 
in the autoencoder_cnn.py file.

The app has 2 canvas, one is a black canvas representing 2D latent space, by clicking on a point on this canvas, we select the coordinates of the latent space for the model
to make his prediction, which is then plotted in the second canvas.
The app is launched with python main.py, but requires latest tensorflow version as well as flask and tensorflowjs. 
It may not work for you if you download the repo, as the goal of this project was merely to a explore VAEs and flask personnaly.
However I will work further to improve it and to make it accesible for users.

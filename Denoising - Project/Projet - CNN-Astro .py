#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 17:29:44 2022

@author: Ahmed Adham

# A noter qu’une partie du code servant à générer le CNN provient du cours du Mike X. Cohen sur la construction des réseaux de neurones 
(A deep understanding of Deep Learning), mais a été très largement modifiée et adaptée
"""



# In[ ]:

"""
 Un problème majeur dans l’observation Astronomique est le bruit généré par les perturbations atmosphériques dans l’image, 
 générant une sorte de « blurring »
 Dans ce projet, nous allons entrainer un CNN à retrouver l’image originale
 Pour cela, nous allons 
	1) prendre des photos de référence, 
	2) générer une série images bruitées à partir d’elles 
	3) construire notre CNN selon les paramètres du papier
 	4) l’entrainer à retrouver des fragments d'images



Le code sera organisé ainsi : 
    A - Fonction de chargement 
    B - Fonction de bruitage des images + bruit Gaussien
    C - Fonction de cropping aléatoire des images 
    D - Génération des images floutées à partir d'images nettes 
    E - Construction de nos images sur lequel le CNN va travailler
    F - Construction du CNN
    G - Entrainement et test
    
"""
# In[ ]:

# Partie 0 : - long à charger la première fois à cause de cv2
# importation des libraries nécessaires au projet 
# Pour le bon fonctionnement, merci de mettre ici le chemin jusqu’au fichier Img_Gif/ (inclus)
Path = '/Users/ahmed/Desktop/Denoising - Project/Img_Gif/' 
# prérequis : Tous les fichiers images se trouvent dans le répertoire « Img_Gif » . 
# Type C:/ XXXX…/Img_Gif/

#######

import numpy as np
import random
import cv2 as cv #pour le calcul du SNR ( pip install opencv-python ) - long à charger la première fois
import imageio #pour expoter nos images facilement
from astropy.convolution import AiryDisk2DKernel #pour la tache d'Airy
from PIL import Image #pour importation des gif
from scipy.signal import convolve2d # pour la convolution


import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')


# pour le CNN 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

### variables nécessaires pour pouvoir lancer le code directement depuis la partie CNN si besoin :
#nombre_de_galaxies = 11
#nombre_de_crop = 2000



# In[ ]:
""" 
A)
    
Créons une fonction chargeant et coupant l'image d'une galaxies, 
"""
def load_Gal_Image ( Nom_de_la_Galaxie, Path):

    Galaxie = Path + Nom_de_la_Galaxie #chemin de l'image 
    imMXX = Image.open(Galaxie) #importation image 
#   imMXX.show() #admirons la galaxie entière...
    imMXXarray = np.array(imMXX) # on convertit l'image en matrice pour Numpy
    size = imMXXarray.shape #voyons la taille de l'image
    size

    return imMXXarray #on renvoie l'image coupée en sortie de la fonction


# In[ ]:
    
""" B)

Nous allons à présent créer une fonction pour bruiter l’image en
#1) effectuant une convolution en 2D sur notre image à l’aide d’une Point-Spread Fonction (PSF) selon la formule décrite page 3 de l’article 
#2) en ajoutant du bruit gaussien avec une STD de 0.01  """


"""# 2.1 D'abord la convolution 2D, Générons une tache d’Airy à l'aide de Astropy """

# On va générer un kernel de PSF correspondant à une tache d’Airy, que l’on va convoluer à notre image initiale. 
# Le kernel aura une Full Width at half maximum à 8 pixels, comme spécifié dans le papier. 
# On peut le mettre à 16x16 pixels de large (au lieu de 64x64) pour gagner en temps de calcul

largeur_tache = 8 #pixels - on peut ajuster plus selon le pouvoir séparateur de notre télescope
kernel_size = 32 #largeur de notre kernel en pixels
airydisk_2D_kernel = AiryDisk2DKernel(largeur_tache,x_size = kernel_size,y_size = kernel_size) #création du Kernel de PSF
airydisk_2D_kernel = np.array(airydisk_2D_kernel) #conversion en tableau numpy

# Voyons à quoi ressemble notre Kernel
plt.imshow(airydisk_2D_kernel, interpolation='none', origin='lower',cmap='Greys',vmax=0.001) 
# vmax=0.001 pour mettre en lumière la forme en "vague" de la tache d'Airy 
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.colorbar()
plt.show()


"""#  écrivons la fonction codant la convolution entre notre Tache d’airy et notre image + ajout de bruit gaussien...    """
def Convolution_Airy (image,kernel,std_noise) :
  convoutput = convolve2d(image,kernel,mode='valid')  #on effectue la convolution entre le kernel et l'image
  gauss = np.random.normal(0,std_noise,(convoutput.shape[0],convoutput.shape[1])) #génération d'une matrice de nombre aléatoires 
  #selon une loi Normale (Gauss), de dimension égale à notre image convoluée
  Noisy_and_Blurry = convoutput + gauss #addition des deux
  
  return Noisy_and_Blurry # on renvoie l'image bruitée et floutée
  


# In[ ]:
""" C)    

Ecrivons la fonction qui découpera nos images en segments de 32x32

""" 

def Cropping (image_nette,image_floue,size_crop) :
    # nos images font 3564*3571
    # Nous allons les « cropper » pour qu’elles fasses "size_crop" x "size_crop" pixels, au hasard
    # nous entrainerons notre réseau à partir de toutes ces petites images
    size = image_nette.shape
    width = size[0] #largeur de l'image originale
    height =  size[1] #hauteur de l'image originale
    
    left = random.randint(size_crop,width-size_crop) #choix d'un pixel(left,top) pour un carré à couper au hasard dans l'image (sans que ca ne dépasse)
    top = random.randint(size_crop,height-size_crop) #choix d'un pixel(left,top) pour un carré à couper au hasard dans l'image (sans que ca ne dépasse)
    right = left + size_crop # on étend ce carré à +size_crop pixels en largeur
    bottom = top + size_crop # on étend ce carré à +size_crop pixels en bas
   
    image_nette_crop = image_nette[left:right,top:bottom] # on coupe
    image_floue_crop = image_floue[left:right,top:bottom] # on coupe

    return image_nette_crop, image_floue_crop

 

# In[ ]: Environ 5 minutes
""" 
D) - Etape prenant quelques minutes, les résultats sont dans le fichier fourni Img_Gif/Noisy 
    - sinon exercuter avec juste 1 galaxie en mettant nombre_de_galaxies à 1, ou en diminuant la taille du kernel de convolution à 16x16 px
    
A l'aide des deux fonctions "charger" et "bruiter" créeons un jeu de données test """


Nom_de_la_Galaxie = ('M31.gif','M51.gif','M81.gif','M101.gif','M104.gif','Hoag.gif','Haltere.gif','Lyre.gif','M24.gif','M45.gif','M1.gif') 
#chargeons pour toutes nos galaxies...
nombre_de_galaxies = 11; # pour étoffer le jeu de données, j'ai ajouté des images de nébuleuses et amas d'étoiles

#on crée une matrice dans laquelle on va mettre nos images vierges sous format np :
Galaxie_np = np.zeros([nombre_de_galaxies,3571,3564])  # 3571,3564 = taille des images en pixel

# On crée une matrice dans laquelle on va mettre nos images bruitées sous format format np
# la matrice contenant le résultat de la convolution est de taille inférieure à l'image non convoluée (
# -> taille image - taille kernel + 1, donc :
Noisy_galaxy = np.zeros([nombre_de_galaxies,3571-kernel_size+1,3564-kernel_size+1]) 


# Chargement des images + bruitage : 
for i in range(nombre_de_galaxies) : 
    Galaxie_np[i] = load_Gal_Image(Nom_de_la_Galaxie[i],Path) #on applique la fonction pour charger et couper l'image
    #Galaxie_np[i] = Galaxie_np[i]/np.amax(Galaxie_np[i])     # Normalisons à présent les données de notre tableau entre 0 et 1, 
    #pour pouvoir utiliser les différentes images pour entrainer notre CNN
    Noisy_galaxy[i] = Convolution_Airy(Galaxie_np[i],airydisk_2D_kernel,0.01)  # on applique notre fonction de bruit + blurring


# On affiche le résultat pour la galaxie n°2 - M51  ; 
fig,ax = plt.subplots(1,2)
ax[0].imshow(Galaxie_np[0],cmap='Greys') # image originale 
ax[0].set_title('Source')
ax[1].imshow(Noisy_galaxy[0],cmap='Greys') # résultat bruité pour la galaxie N°2
ax[1].set_title('Après convolution')
plt.show()
# 

    # enregistrons nos images floutées pour les sauvegarder avant d'avancer 
for i in range(nombre_de_galaxies) :
    Noisy_galaxy[i] = Noisy_galaxy[i]*255/np.amax(Noisy_galaxy[i]) # on normalise entre 0 et 255
    nom = Path + '/Noisy/' + 'No' + str(i) + '.png'
    imageio.imwrite(nom,Noisy_galaxy[i]) #choisi au lieu de plt.imsave car plus esthétique : l'image est encodée en noir et blanc ! 


# nos images nettes ne sont pas comparables aux floues en tant que telles puisquela convolution a "élagué" les bords des images floues.
# Nous allons donc couper le bord des images nettes et les enregistrer également en supprimant des lignes et des colonnes tout autour 
Original_galaxy = np.zeros([nombre_de_galaxies,3571-kernel_size+1,3564-kernel_size+1])  
# Initialisation de la matrice qui aura les images vierges de la même taille que les images floutées
 

for i in range(nombre_de_galaxies):
    Original_galaxy[i]= Galaxie_np[i,15:3571-16,15:3564-16] #on raccourcit l'image originale de 16 à gauche , à droite en haut et en bas (-32 au total)
    # Bien changer 15/16 si modification taille du kernel de convolution
    nom = Path + '/Original/' + 'Or' + str(i) + '.png'
    imageio.imwrite(nom,Original_galaxy[i]) #choisi au lieu de plt.imsave car plus esthétique : l'image est encodée en noir et blanc ! 


""" A noter qu'elles sont déjà pré-enregistrées dans le fichier Img_Noisy """

for i in range(nombre_de_galaxies):
    print(cv.PSNR(Original_galaxy[i], Noisy_galaxy[i])) #Calcul du Peak SNR : de base autour de 32 pour chaque image

# Pour une PSF de 8px :
# 1 32.25 dB  
# 2 35.13 dB   
# 3 33.60 dB  
# 4 34.25 dB  
# 5 32.21 dB  
# 6 32.84 dB  

# Pour une PSF de 16px :
# 1 29.71 dB  
# 2 32.92 dB   
# 3 31.59 dB  
# 4 31.59 dB  
# 5 29.43 dB  
# 6 30.50 dB    


# In[ ]: Construction de la base d'images
""" E) 

A présent que l'on a nos images floutées et bruitées, Nous allons constituer une base d’images nettes 
et une bases d’images floues correspondante, afin d’entrainer notre réseau de neurone par la suite. 
(Ce processus est assez court malgré le nombre important de fichiers).
    
""" 

# Nous avons donc Original_galaxy et Noisy_galaxy en grande image.

nombre_de_crop = 2000 # 2000 crops * 11 galaxies = 22000 images pour travailler - pour des soucis de temps calcul je ne met pas 
# d'augmentation artificielle (rotation, ajout de bruit, etc...)
taille_du_crop = 32 

for i in range(nombre_de_galaxies) :    #boucle sur le nombre_de_galaxies
   for j in range(nombre_de_crop) :   #boucle sur le nombre de crop
       im_nette,im_floue =   Cropping(Original_galaxy[i],Noisy_galaxy[i],taille_du_crop) # on prend un carré de 32x32 aléatoire des deux images

       nom = Path + '/Data_Set/' + str(i) + '_' +  str(j) + '_Ori.png' # on assigne l'image net à son chemin
       im_nette_ui8 = (im_nette).astype(np.uint8) 
       #imageio.imwrite(nom,im_nette_ui8) #on enregistre l'image nette - a noter que certaines fois u
       im = Image.fromarray(im_nette_ui8)
       im.save(nom) # on enregistre les images coupées


       nom1 = Path + '/Data_Set/' + str(i) + '_' + str(j) + '_Noi.png' # on assigne l'image floue à son chemin
       im_floue_ui8 = (im_floue).astype(np.uint8)
       #imageio.imwrite(nom1,im_floue_ui8) #on enregistre l'image floue 
       im = Image.fromarray(im_floue_ui8)
       im.save(nom1) # on enregistre les images coupées
        

    
""" E-bis) Créons une fonction pour évaluer le Peak Signal to Noise Ratio moyen entre nos images 
- nous la réutiliserons pour évaluer notre réseau de neurones à la fin
"""
n = 0
psnr = np.zeros(nombre_de_galaxies*nombre_de_crop) # initiation d'une matrice de valeurs
for i in range(nombre_de_galaxies):
    for j in range(nombre_de_crop): # on fait le tour des psnr en chargeant les images unes a unes et en le calculant
        nom = Path + '/Data_Set/' + str(i) + '_' +  str(j) + '_Ori.png' 
        nom1 = Path + '/Data_Set/' + str(i) + '_' + str(j) + '_Noi.png' 
        img1 = Image.open(nom) #Original
        img2 = Image.open(nom1) #Noisy
        psnr[n] = cv.PSNR(np.array(img1), np.array(img2)) #Calcul du Peak SNR
        n = n+1

print(np.mean(psnr,0)) # sur mon essai : pSNR : 33dB entre net et flou
print(np.nanvar(psnr,0)) # variance sur mon essai : 29
print(np.nanstd(psnr,0)) # écart-type 5.3


# en allante dans le répertoire /Data_Set/ on peut observer les images sur lesquelles le CNN va travailler 

# In[ ]: Construction du CNN
""" F 
A présent que nos jeu d'images a été généré, construisons notre CNN 
- A noter que cette section est inspirée du Cours sur le Deep-Learning Sus-cité

On va dans un premier temps :
    - 1) Charger nos données dans une matrice 
    - 2) Creation du jeu de données Train-Set /// Test-Set
""" 

# In[ ]: 1) Création des matrices de données 

# Il est conseillé de faire "%reset" des variables ici pour désengorger la RAM
# variables nécessaires pour la suite : 
# Path = '/Users/Ahmed/Desktop/Denoising - Projet /Img_Gif' 
# nombre_de_galaxies = 11
# nombre_de_crop = 2000

#Chargement des données à partir des images
# initiation des matrices d'images nettes et floues
n = 0
labels = np.zeros([nombre_de_galaxies*nombre_de_crop,32,32])
data = np.zeros([nombre_de_galaxies*nombre_de_crop,32,32])

for i in range(nombre_de_galaxies):
    for j in range(nombre_de_crop-1): # on fait le tour des psnr en chargeant les images unes a unes et en le calculant
        nom = Path + '/Data_Set/' + str(i) + '_' +  str(j) + '_Ori.png' 
        nom1 = Path + '/Data_Set/' + str(i) + '_' + str(j) + '_Noi.png' 
        labels[n] = np.array(Image.open(nom)) #Original
        data[n] = np.array(Image.open(nom1)) #Noisy
        n = n + 1

        
# On va normaliser les valeurs de nos jeux de données entre 0 et 1 
dataNorm = data / np.max(data) 
labelsNorm = labels / np.max(labels)


im_2_plot = random.randint(0,1000); # chiffre au hasard pour choisir une image du jeu de données
fig,ax = plt.subplots(1,2)
ax[0].imshow(labels[im_2_plot],cmap='Greys') #plot image d'origine
ax[0].set_title('Originale')
ax[1].imshow(data[im_2_plot], interpolation='nearest',cmap='Greys') #plot image floutée/bruitée
ax[1].set_title('Floutée')
plt.show()
# -> l'objectif sera de constituer l'image normale à partir de la floutée

# on reshape notre jeu d'images non bruitées pour correspondre au format de la sortie du CNN
# en effet il ne donne pas une matrice de 32*32 mais une matrice de 1x1024
labelsNorm = labelsNorm.reshape(labelsNorm.shape[0],1024) # Labels = données nettes

dataNorm = dataNorm.reshape(dataNorm.shape[0],1,32,32) # Data = données floues
#reshape des donnes sur 12000 images * 1 canal (gris) * 32 * 32, important pour le CNN



# In[ ]: 2) Constituion du training set et test set

# 1 : conversion en tenseurs pour être utilisés par torch
dataT   = torch.tensor( dataNorm ).float()
labelsT = torch.tensor( labelsNorm ).float()

# 2 : utilisation de scikitlearn pour couper le jeu de données, avec 90% pour l'entrainement et 10% pour l'évaluation
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labelsT, test_size=.1)

# 3 : Conversion en PyTorch Datasets
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

# 4 : création des objets dataloader, en vue de faire des minibaatch : on alimente notre CNN d'une "vollée" (minibatch)
# de plusieurs données dont on moyenne ensuite les coeffiscients de backpropagation pour éviter des effests
# où une donnée érronnée loin du reste vienne fausser l'apprentissage
batchsize    = 50 #50 dans le papier, 

train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True) 
#utilisation de la fonction DataLoader, avec Shuffle activé pour mélanger les données, et créer le Train set 
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0]) 
#utilisation de la fonction DataLoader, avec Shuffle activé pour mélanger les données, et créer le Test set

# Vérification de la taille de notre jeu de données : 
train_loader.dataset.tensors[0].shape



# In[ ]:
""" G 

A présent entrons dans le coeur du CNN...

"""

def GalaxyNet(printtoggle=False):

  class GalaxyNet_class(nn.Module):
    def __init__(self,printtoggle):
      super().__init__()

      ### convolution layer n° 1
      self.conv1 = nn.Conv2d(1,64,kernel_size=10,stride=1,padding='valid')
      # 1 (car 1 entrée d'échelle de gris), 
      # 64 kernels pour la couche 1 (dans le papier)
      # ... de taille 10x10
      # stride = 1 pour que le kernel "avance" de 1px par 1 px
      # padding -> pas de padding, désactivé (choix du papier) - 'valid' = padding off

# taille "Output size" : (32-10+0)/1 + 1 = 23 (comme spécifié dans le papier)
# formule ((n entrée - taille kernel + 2*taille du padding)/stride +1 )

      self.conv2 = nn.Conv2d(64,16,kernel_size=6,stride=1,padding='valid')
      # 64 (car 64 entrée depuis la couche précédente - 1 par kernels), 
      # 16 kernels pour la couche 2 (dans le papier)
      # ... de taille 6x6
      # stride = 1 pour que le kernel "avance" de 1px par 1 px
      # padding -> pas de padding désactivé (choix du papier) - 'valid' = padding off

# taille "Output size" : (23-6+0)/1 + 1 = 18 (comme spécifié dans le papier après la couche 2)


      self.conv3 = nn.Conv2d(16,16,kernel_size=5,stride=1,padding='valid')
      # 16 (car 16 entrée depuis la couche précédente, 1 par kernel), 
      # 16 kernels pour la couche 2 (dans le papier)
      # ... de taille 5x5
      # stride = 1 pour que le kernel "avance" de 1px par 1 px
      # padding désactivé - 'valid' = padding off

# taille "Output size" : (18-5+0)/1 + 1 = 14 (comme spécifié dans le papier)

      # combien d'éléments arrivent à la fin avant notre sortie ?
      expectSize = np.floor( (14+2*0-1)/1 ) + 1 #nombre de lignes : 14
      expectSize = 16*int(expectSize**2) # 14*14 : nombre de pixels de sortie * le nombre de Kernels
      # sortant de la couche 3 (il y en a 5) -> 14 * 14 * 16 = 3136
      
      ## on rajoute un fully-connected layer avant la sortie, linéaire, mais pas précisé dans le papier
      self.fc1 = nn.Linear(expectSize,1024) # 3136->1024 neurones, non précisé dans le papier

      ### sortie
      self.out = nn.Linear(1024,1024) # 32*32 sorties

      # afficher la taille des tenseurs 
      self.print = printtoggle

    # forward pass 
    def forward(self,x):
      
      print(f'Input: {x.shape}') if self.print else None
       # convolution  -> relu
      x = F.relu(F.max_pool2d(self.conv1(x),1))  # prend les données de conv1 et applique une non linéarité (ReLu)
      print(f'Layer conv1/pool1: {x.shape}') if self.print else None # afficher pour chaque couche l'architecture

      # and again: convolution  -> relu
      x = F.relu(F.max_pool2d(self.conv2(x),1))   # prend les données de conv2 et applique une non linéarité (ReLu)
      print(f'Layer conv2/pool2: {x.shape}') if self.print else None # afficher pour chaque couche l'architecture

      # and again: convolution -> relu
      x = F.relu(F.max_pool2d(self.conv3(x),1))  # prend les données de conv3 et applique une non linéarité (ReLu)
      print(f'Layer conv3/pool3: {x.shape}') if self.print else None # afficher pour chaque couche l'architecture
 
      # reshape pour que les tailles matchent avec le linear layer (qui adment pas d'entrées à N-D)
      nUnits = x.shape.numel()/x.shape[0]
      x = x.view(-1,int(nUnits))
      if self.print: print(f'Vectorize: {x.shape}')
      
      # linear layer avant la sortie en Fully-Connected Network
      x = F.relu(self.fc1(x))
      if self.print: print(f'Layer fc1: {x.shape}')
      
      x = self.out(x) # sortie
      if self.print: print(f'Layer out: {x.shape}')

      return x
  
    """
On a en sortie le CNN tel que désigné dans le papier : 
    
Input: torch.Size([128, 1, 32, 32]) Input
Layer conv1/pool1: torch.Size([128, 64, 23, 23]) Couche 1
Layer conv2/pool2: torch.Size([128, 16, 18, 18]) Couche 2
Layer conv3/pool3: torch.Size([128, 16, 14, 14]) Couche 3
Layer out: torch.Size([128, 196])
"""
  
  # créons notre nom de modèle : net
  net = GalaxyNet_class(printtoggle)
  
  # function de perte
  lossfun = nn.MSELoss() #fonction d'évaluation , non précisée dans le papier mais Cross Entropy adaptée pour 
  # les comparaisons d'images 

  # optimizer
  optimizer = torch.optim.Adam(net.parameters(),lr=0.001) 
  # J'ai pris un Adam optimiser car je n'ai pas réussi à faire fonctionner avec l'optimizer proposé dans le papier (vanishing gradient ?) 
 
  # optimizer = torch.optim.SGD(net.parameters(),lr = 0.01, momentum = 0.9, nesterov = 'True', weight_decay = 0.01) 
  #proposé par le papier - ajout d'un weight_decay = 0.005 pour la régularization , 
  # vu que le CNN semble parfois mémoriser, mais SGD ne marche pas très bien : au mieux image qui se reproduit (mémorisée par le CNN) , ou bruit

  return net,lossfun,optimizer #renvoie les caractéristiques du réseau en sortie


# In[ ]: mettre en # si on veut rester sur la CPU, sinon accélération GPU !  -  
# Chez moi iMac avec carte graphique AMD donc Cuda indsponible: non testé
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
print(device)

#GalaxyNet.to(device) # enoyer réseau à la GPU
#dataNorm = dataNorm.to(device) # enoyer variables à la GPU
#labelsNorm = labelsNorm(device) # enoyer variables à la GPU

#Output = GalaxyNet(data)

#( Si c'est activé il faut théoriquement chercher les variables de la GPU à la partie affichage tout en bas )

# In[ ]:


# Définition de la fonction qui va entrainer le réseau

def function2trainTheModel():

  # nombre d'épochs d'entrainement
  numepochs = 1
  
  # réaction d'un nouveau modèle
  net,lossfun,optimizer = GalaxyNet()

  # initialisation de toutes les variables de mesure de perte et accuracy
  losses    = torch.zeros(numepochs)
  trainAcc  = []
  testAcc   = []


  # loop sur les épochs : 
  for epochi in range(numepochs):

    
    net.train() # on met notre net sur le mode entrainement -> backpropagation et dropout activée
    batchAcc  = []
    batchLoss = []
    for X,y in train_loader:

      yHat = net(X) # X = données floutées, yHat = sortie après avoir mit X dans notre "net"
      loss = lossfun(yHat,y) # evaluation de la fonction de perte

      # fonctions de backpropagation :
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # résultat de la fonction de coût
      batchLoss.append(loss.item())

      # Evaluation de la précision à travers le  PSNR
      
      psnr = cv.PSNR( y.cpu().detach().numpy(), yHat.cpu().detach().numpy()) # Calcul du Peak SNR sur chaque image de notre jeu de données
      #y et yHat (originale et sortie du CNN) - cpu().detach().numpy() car tenseur -> numpy-array
      batchAcc.append( psnr ) # on renvoie la valeur de notre PSNR sur le batch

     
    # Moyenne de la précision moyenne sur les batchs
    trainAcc.append( np.mean(batchAcc) )

    # Moyenne des pertes moyennes sur les batchs
    losses[epochi] = np.mean(batchLoss)

    net.eval()     # Evaluation du réseau sur le Test set (10% des données)
    X,y = next(iter(test_loader)) # extraire X,y du test dataloader
    with torch.no_grad(): # desactive gradient auto,  et le dropout
      yHat = net(X) #on passe notre test_set flou X sur le jeu 
      
    # et on calcule sur les résultats en sortie le psnr
    psnr_1 = cv.PSNR( y.cpu().detach().numpy(), yHat.cpu().detach().numpy()) # Calcul du Peak SNR sur notre jeu de données
    testAcc.append(psnr_1)
    
    
  return trainAcc,testAcc,losses,net #on renvoie ces variables pour les afficher


# 
# In[ ]:
""" LONG !!!!!!! Compter dans les 1 minute par epoch : 

Sur iMac 2017 i5 3.4GHz - sur CPU Intel i5 : 57 secondes par epoch
GPU AMD : Cuda Impossible

Je conseille de ne lancer que sur quelques époch (8 permettent déjà de se rendre compte de l'effet') 
et charger (plus bas) le résultat à 20 ou à 40 Epochs que j'ai fourni dans le fichier pours les tests
"""
    
# on lance le modèle 
trainAcc,testAcc,losses,net = function2trainTheModel()


# In[ ]: Performances de ce modèle et de son apprentissage

fig,ax = plt.subplots(1,2,figsize=(16,5))

ax[0].plot(losses,'s-')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Model loss')

ax[1].plot(trainAcc,'s-',label='Train')
ax[1].plot(testAcc,'o-',label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('PSNR (cB))')
ax[1].set_title(f'Final model pSNR : {testAcc[-1]:.2f}dB')
ax[1].legend()

# In[ ]: Sauver notre modèle

torch.save(net.state_dict(),'trained_model.pt') #sauvegarder notre réseau et ses poids

# In[ ]: charger un modèle pré-enregistré 

# trained_model_20 = 20 épochs d'apprentissage

model_saved = GalaxyNet()[0] #si on execute le code depuis ici, lancer la partie au dessus qui définit le modèle
#model_saved.load_state_dict(torch.load('trained_model.pt')) #charger un modèle déjà entrainé sauvegardé
model_saved.load_state_dict(torch.load('trained_model_20.pt')) #charger le modèle que j'ai entrainé à 20 epochs


# In[ ]: Afficher les images
""" pour afficher les images. Possibilité de charger un modèle pré-entrainé """

X,y = iter(train_loader).next()

with torch.no_grad(): # deactivates autograd
    yHat = net(X) #si on veut visualiser les données du modèle actuel
    #yHat = model_saved(X) #si on veut visualiser les données d'un modèle enregistré

y_np_1 = y.cpu().detach().numpy()
yHat_np_1 = yHat.cpu().detach().numpy() 
X_np_1 = X.cpu().detach().numpy() 

for im_2_plot in range(50): #50 = nombre d'images qu'on veut regarder
    y_np = np.reshape(y_np_1[im_2_plot],(32,32))
    X_np = np.reshape(X_np_1[im_2_plot],(32,32))
    yHat_np = np.reshape(yHat_np_1[im_2_plot],(32,32))

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(y_np,cmap='Greys') #plot image d'origine
    ax[0].set_title('Originale - y')

    ax[1].imshow(X_np,cmap='Greys') #plot image d'origine
    ax[1].set_title('Floutée - X ')

    ax[2].imshow(yHat_np,cmap='Greys') #plot image d'origine
    ax[2].set_title('Floutée reconstruite -yHat')
    plt.show()
   
   # on constate que certaines images sont reconstruites 


psnr = cv.PSNR( y.cpu().detach().numpy(), yHat.cpu().detach().numpy()) # Calcul du Peak SNR sur notre test-set :
# on est passé de 33dB avant entrainement à 77dB après


""" 

Fin du papier


"""


# In[ ]: Reconstruire une image entière 

"""
Possible de lancer le code directement depuis cet emplacement ,
juste exectuer la cell#6  (pour charger le modèle du CNN)
et réactualiser la variable du chemin Path


Nous allons maintenant charger des images, les couper en morceaux de 32*32px qui seront injectés dans notre net et nous reconstruirons la sortie.
pour éviter un effet de "petits carrés posés à côté" on va reconstruire l'image plusieurs fois en décallant un peu le ciseau à chaque fois puis en moyennant le tout
cela sur 4 pixels dans la longueur et 4 pixels en largeur 

Ceci prend quelques minutes, 
"""

# pour relancer le code d'ici directement, il faut executer la cellule 6 qui initialise le modèle 
# runcell('[ ], #6', '/Users/ahmed/Desktop/Denoising - Project/Projet - CNN-Astro .py')

# pour faire à partir d'un modèle enregistré 
model_saved = GalaxyNet()[0]
model_saved.load_state_dict(torch.load('trained_model_20.pt')) #charger le modèle que j'ai entrainé à 20 epochs


chemin = Path + 'Jamais_vues/' +  'Jupiter.png' #  chargement de Jupiter du début
#chemin = Path + 'Jamais_vues/' + 'Nb-Trefle-Bruitee.png' # avec une image qu'il connait

x = cv.imread(chemin,0)
x = np.array(x) 

#x = range(320*320)
#x = np.reshape(x,(320,320))
#print (x)

x = x/np.amax(x) #normaliser à 1

taille_du_crop = 32 

n_cases_ligne = round(x.shape[0] / taille_du_crop) - 2 # nombre cases de 32*32 sur les lignes round au cas où il y a un nombre impair de pixels
# - 2 car c'est pas toujours un multiple de 32, on coupe la fin (2 et non 1 car on va faire une fenêtre glissante et reconstruire 32 fois l'image à 1 px de décallage à chaque fois)
n_cases_colonne = round(x.shape[1] / taille_du_crop) - 2 # nombre cases de 32*32 sur les colonnes 
n_carres = n_cases_colonne*n_cases_ligne # n cases total


#n_images,dimension RGB (gris -> 1), taille pixels

# on définit une fonction qui coupe en carrées de 32x32 successif
def crop (x,taille_du_crop,n_cases_ligne,n_cases_colonne,n_carres,shift_x,shift_y):
    crop_image = np.zeros([n_carres,32,32]) # initialisation des matrices 
    n = 0
    for i in range(n_cases_ligne):
        for j in range(n_cases_colonne):
            crop_image[n] = x[i*taille_du_crop+shift_x:i*taille_du_crop+taille_du_crop+shift_x,j*taille_du_crop+shift_y:j*taille_du_crop+taille_du_crop+shift_y]
           #shift_y et shift_x car on va reconstruire puis refaire en décallant le crop pour éviter d'avoir un effet de carrées posées à côtés, et moyenne le tout
            n = n+1
    return crop_image


# on définit une fonction de reconstruction des images, qui prend en entrée toutes les images de 32x32 et sort les images reconstruite en 32x32 
def reconstruct (crop_image,x,n_carres):
    crop_image = crop_image.reshape(crop_image.shape[0],1,32,32) # reshape en 
    #n_images,dimension RGB (gris -> 1), taille pixels
    # on ajoute les images coupees à notre modèle  
    # plt.imshow(crop_image[1,0,:,:], interpolation='nearest') # à quoi ca ressemble
    tenseur_image_coupee = torch.from_numpy(crop_image).float() #converti carré de 32x32 en tenseur et float       
    resultat = model_saved(tenseur_image_coupee)
    # on passe notre carré dane le modèle
    resultat = np.reshape(resultat.cpu().detach().numpy(),(n_carres,32,32))
    # on le ramène en tableau np
    return resultat


resultat = np.zeros(x.shape)
new_image = np.zeros(x.shape)  # image de sortie de même taille qu'image en entrée


max_shift_x = 3
max_shift_y = 3
image_mean = np.zeros([x.shape[0]+max_shift_x,x.shape[1]+max_shift_y])

for i in [0,1,2,3]: #Bien corriger le max_shift_x à la valeur max que peut prendre i, pour régler la taille des matrices
    for j in [0,1,2,3]: #Bien corriger le max_shift_y à la valeur max que peut prendre i, pour régler la taille des matrices
        crop_image = crop(x,taille_du_crop,n_cases_ligne,n_cases_colonne,n_carres,i,j) # on crop
        resultat_int = reconstruct (crop_image,x,n_carres) # on reconstruit # environ 8 seconde
        
        image_mean[0+i:x.shape[0]+i,0+j:x.shape[1]+j] = image_mean[0+i:x.shape[0]+i,0+j:x.shape[1]+j] + new_image

        n = 0
        for k in range(n_cases_ligne):
            for l in range(n_cases_colonne):
                new_image[k*taille_du_crop:k*taille_du_crop+taille_du_crop,l*taille_du_crop:l*taille_du_crop+taille_du_crop] = resultat_int[n]
                # on écrit sur la nouvelle image le résultat
                n = n+1
                


image_mean = image_mean*255/np.amax(image_mean) # on renormalise 

fig,ax = plt.subplots(1,2)
ax[0].imshow(-x,cmap='Greys') # image floue 
ax[0].set_title('Originale')
ax[1].imshow(-image_mean,cmap='Greys') # image "reconstruite"
ax[1].set_title('Après convolution et "reconstruction"')
plt.show()

""" Fin """


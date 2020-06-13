# **A DCGAN Project to Generate Anime Faces**
## 👉 Contributors: 👨‍💻 **Nirdesh Dwa** and 👩‍💻 **Priti Zimba** 
### 🙏 Our sincere thanks to **Sushil Thapa** and **Kshitiz Mandal**
---
### **References:**

* [Google's GAN Page](https://developers.google.com/machine-learning/gan)
* [DCGAN Tensorflow](https://www.tensorflow.org/tutorials/generative/dcgan)
* [Tricks for GANs](https://lanpartis.github.io/deep%20learning/2018/03/12/tricks-of-gans.html)
* [DCGAN for 🐱 Image Generation](https://github.com/simoninithomas/CatDCGAN)
* [DCGAN for 🐶 Image Generation](https://towardsdatascience.com/dcgans-generating-dog-images-with-tensorflow-and-keras-fb51a1071432)
* [GAN HACKS used label smoothing and noise](https://mc.ai/how-to-implement-gan-hacks-to-train-stable-generative-adversarial-networks/)
* [GAN Loss Function](https://machinelearningmastery.com/generative-adversarial-network-loss-functions/)
* [Comparing Loss Functions for GAN](https://medium.com/@tayyipgoren/keras-optimizers-comparison-on-gan-b8b98c3d8645)
* [ADAM Optimizer](https://stackoverflow.com/questions/37842913/tensorflow-confusion-regarding-the-adam-optimizer)
* [Understaning Gradient Tape](https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/)
* [Guide to GAN Failure](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)

----

## 👉 **[SUMMARY OF GAN PAPER BY ME](https://drive.google.com/file/d/1XwiY8gXDmyCQTIMbodKekD4ALAuLThgg/view?usp=sharing)**
## 👉 **[LINK TO GOOGLE COLAB FOR THIS PROJECT](https://colab.research.google.com/drive/13URXit4-Qk4JtX-9H71loQPiNf6k2AlG?usp=sharing)**

----
## **Downloading and running the code**
I recommend using 👉 [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) 

Virtual environment is an isolated space for your project. It makes managing dependencies, depoyment and sharing of code whole lot easier. It basically does two things:
1. Keep the dependencies of each project isolated from eachother.
2. Make sure your install dependencies will affect the system.

**Install Virtual Environment **

Note: If you are using python3, you should already have venv module.
````
$ pip install virtualenv
````
**Creating Virtual Environment**

Note: Enter the anime_dcgan inside AnimeGAN directory and run the code
````
# Python 2:
$ virtualenv animegan_env

# Python 3
$ python3 -m venv animegan_env
````
**Activate the virtual environment**
````
$ source animegan_env/bin/activate
````

**Stop using the virtual environment**

Note: Do not do it now, do it when you have completed the task and need to stop the virual environment
````
$ deactivate
````

**Install the required packages using**
````
$ pip3 install -r requirements.txt
````
**Run app.py**
````
$ python app.py
````

**Find the served template**

Go to the flollowing location on your browser
````
localhos:5000
````
## **Output Images and GIFs**
**CHERRY PICKED SAMPLES**

![CHERRY PICKED SAMPLES](https://i.ibb.co/5GJf8WM/cherry-picked-images.jpg)

**GIF FROM TRAINING 30 EPOCHS**

![GIF FROM TRAINING 30 EPOCHS](https://media.giphy.com/media/lpt5fsbt9Qg2Xg5xO7/source.gif)

**LOSS PER EPOCHS TRAINING 30 EPOCHS**

![LOSS PER EPOCHS TRAINING 30 EPOCHS](https://i.ibb.co/CBcCxMJ/Screen-Shot-2020-06-12-at-04-13-43.png)

**GIF FROM TRAINING 280 EPOCHS**

![GIF FROM TRAINING 280 EPOCHS](https://media.giphy.com/media/QU3KEhA6Y1Wt07YDFS/source.gif)

**LOSS PER BATCH TRAINING 280 EPOCHS**

![LOSS PER BATCH TRAINING 280 EPOCHS](https://media.giphy.com/media/j3hTZFk7Y1s3rAwq8u/source.gif)


----
## **CODE WALKTHROUGH**
> 1. Most of the code are custom written, credit has been given to codes that are copied from other sources or modified from other sources.
2. This is our first GAN as well as Tensorflow project, so the code might not look professional, however, we have tried out best to make it easy for anyone who finds the notebook easy to run the code.
3. We have also mentioned all the problems we faced during the development to make it easy for beginners.












## **IMPORTING REQUIRED LIBRARIES** 


> 1. Note: Matplotlib.imread imports png as [0-1] float while jpg as [0-255] int. It is a good idea to use CV2 instead of consistency.
2. Gridspec is used in this notebook for removing the padding between the subplots in pyplot
3. display is used to clear output in the code
4. shutil is used for working with files
5. plot_model for plotting the summary with states of a model





```
import os#working with the file system
import tensorflow as tf
import numpy as np
from glob import glob #for working with files
import pickle as pkl


import cv2
import matplotlib.pyplot as plt
import shutil #for working with files
from google.colab import files #for uploading the file into google drive
from google.colab import drive #for connecting with google drive
from matplotlib import image
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import ImageGrid #for showing images in grid
import time
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.compat.v1 import truncated_normal_initializer 


%matplotlib inline
import PIL
from PIL import Image
import matplotlib.gridspec as gridspec


import imageio#for displaying animation

import zipfile

import IPython
from IPython import display
```

## **Find GPU**
The following code helps us find GPU device name. If no GPU is available, the code will throw an error message, you can neglect the message.


> Google Colab provides GPU runtime for 12 hrs per day for free users, so it's a good idea to write the code without GPU and change to GPU runtime during the training.






```
# check if GPU is being used
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

    Found GPU at: /device:GPU:0


## **Connecting Google Drive** 


1.  Import drive from colab
2.  Use drive.mount('content/gdrive')  to mount the drive
3.  Click on the link provided
4.  Login to your account
5.  Copy the authorization key after logging in
6.  Paste in the input field and press enter
7.  You'll get an acknowledgment after the drive is mounted


> Mount drive in '/content/drive'

> Drive can be accessed from the folder icon on the left or '/content/drive/My Drive'






```
drive.mount('/content/gdrive') #connect gdrive
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).


## **Defining all the necessary paths**


1.   absolute_path: base path for all the folders
2.   dataset_path: the pate of the zip file downloaded from kaggle to make sure we donot redownload it if it exists.
3.   data_folder: folder where the downloaed dataset will be extracted
4.   resize_folder: folder where the extracted data will be resized and stored.
5.  sampled_folder: In case of initial tryout, it is hard to train model on all the images, this folder can be used to move few images from the main folder and train on those partial set of images
6. array_folder : Save and load raw_images.npy and scaled_images.npy so that we don't have to laod and scale data multiple times
checkpoint_folder =  project_folder + 'checkpoints/'
output_folder = project_folder + 'generated_images/'
model_folder = project_folder + 'models/'
model_summary_folder = project_folder + 'models_summary/'
discriminator_folder = model_folder+'discriminator/'
generator_folder = model_folder+'generator/'
plot_folder = model_folder+'plot/'
animation_folder = model_folder+'animation/'




```
absolute_path = '/content/gdrive/My Drive/'
project_folder = absolute_path + 'AnimeGAN/'
dataset_path = project_folder+'anime-faces.zip'
data_folder = project_folder + 'data/'
resized_folder = project_folder + 'resized_data/'
sampled_folder = project_folder + 'sampled_data/'
array_folder = project_folder + 'saved_arrays/'
checkpoint_folder =  project_folder + 'checkpoints/'
output_folder = project_folder + 'generated_images/'
model_folder = project_folder + 'models/'
model_summary_folder = project_folder + 'models_summary/'
discriminator_folder = model_folder+'discriminator/'
generator_folder = model_folder+'generator/'
plot_folder = model_folder+'plot/'
animation_folder = model_folder+'animation/'
generated_sample_folder = model_folder+'generated_sample/'
```

## **Defining Necessary Functions for working with Directories and Files**


1.   **create_dir()**: Helps to create directory if does not exist, *args:path*
2.   **move_files()**: Helps to move files from one directory to another, might be required in case of moving sample images
3.   **is_empty()**:Helps to see if a directory exists and is empty








```
#functon to create folder in the given path if not exists
def create_dir(path):
  try:# Create target Directory if not exists
    os.mkdir(path)
    print("Directory " , path ,  " Created ")
    return True
  except FileExistsError:
    print("Directory " , path ,  " already exists")

#function to check if the directory exist and is not empty
def is_empty(path):
  if os.path.exists(path):
      print('Directory exists')
      if sum([len(files) for r, d, files in os.walk(path)]) > 0:
        print('Directory is not empty')
        return False
      else:
        print('Directoy is empty')
        return True   
  else:
    print('Directory does not exist') 
    return True



#function for moving files
def move_files(source,dest,num_of_files_to_move=None):
  if os.path.exists(dest) and sum([len(files) for r, d, files in os.walk(dest)]) > 0:
    
      print('Aborting, the destination folder is not empty')
      return
  if not os.path.exists(dest):     
    os.mkdir(dest)
    print(f'Directory created at {dest}')
  count=0
  files = os.listdir(source)
  for f in files:
    print(f'{source+f} -> {dest+f}')
    count=count+1
    shutil.move(source+f,dest)
    if num_of_files_to_move is not None:
      if (count>num_of_files_to_move): 
        break
    print(f'{count} files moved')
```

## **Defining Necessary Functions for working processing Images**


1.   **resize_images()**: Resize images and move to a new folder, *args:path*
2.   **normalize_images()**:Scale image pixes between [-1 to 1] if centered or between [0-255]
3.   **unnormalize_images()**:Scale image back to original pixels







```
#function for resizing files
def resize_images(source,dest,num_of_files_to_resize = None):
  if os.path.exists(dest) and sum([len(files) for r, d, files in os.walk(dest)]) > 0:
      print('Aborting, the destination folder is not empty')
      return
  if not os.path.exists(dest):     
    os.mkdir(dest)
  count = 0
  for each in os.listdir(source):
    image = cv2.imread(os.path.join(source, each))
    image = cv2.resize(image, (64, 64))
    cv2.imwrite(os.path.join(dest, each), image)
    count +=1
    if num_of_files_to_resize is not None:
      if (count>num_of_files_to_resize): #for moving 7698 files
        break
  print(f'{count} files from {source} resized and saved at {dest}')


def normalize_images(raw_images,centered=True):#scale the feature between (-1,1), images is array created above
  images = np.array(raw_images)
  if centered:
    images = (images - 127.5)/127.5
  else:
    images = images/255.0
  return images #we need it to rescale the feature back

#scale back to the actual pixel value range for the model outputs
def unnormalize_images(scaled_array,centered=True):
  if centered:
    scaled_array = (scaled_array*127.5)+127.5
  else:
    scaled_array = scaled_array*255.0
  return scaled_array.astype(np.uint8) #always convert to intvalue for imshow to display integer (can show 0-1 floa to 0-22 int not float)
```

## **Saving and loading arrays**

1. **load_images()**: load image files and save to array

1.   **save_array()**: save array or optionally apply a function passed to the array and save array
2.   **load_array()**: If the array is found load array or create an array and save iter



```
#load images and convert to numpy array
def load_images(path):
  IMAGES = []#the array we are going to populate with image data
  for filename in os.listdir(path):
    img_data = np.asarray(Image.open(path+filename))
    IMAGES.append(img_data)
    print('> loaded %s %s' % (filename, img_data.shape))
  return IMAGES #returns array of image data


#save arrays to to load directly
def save_array(filepath,array,datafunc=None):
  if datafunc != None:
    array = datafunc(array)
  np.save(filepath, np.asarray(array))
  print(array.shape)
  return array


def load_array(path,filename, data, datafunc = None):
  if os.path.exists(path+filename):
    print('Backup file exists. Loading from the backup file.')
    return(np.load(path+filename))
  else:
    print('Backup not found. Creating array and saving to a file.')
    if not os.path.exists(path):
      os.mkdir(path)
    return(save_array(path+filename,data,datafunc)) 

```

## **Displaying Images**


1.   **save_array()**: save array or optionally apply a function passed to the array and save array
2.   **load_array()**: If an array is found load array or create an array and save iter



```
# Displaying Images in a grid
def display_images(image_array,start_pos = 0, cols=4, rows=4,fig_size= (10., 10.),grid=222,pad=0): #show images
  # plt.clf()
  fig = plt.figure(figsize=fig_size)
  grid = ImageGrid(fig, grid,  # similar to subplot(111)
                  nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                  axes_pad=pad,  # pad between axes in inch.
                  )
  
  
  for ax, im in zip(grid, image_array[start_pos:(start_pos+rows*cols)]):
      # Iterating over the grid returns the Axes.
      ax.imshow(im)

plt.show()
```


```
# create the project_folder if not exists
create_dir(project_folder)

# change the working directory to project_directory
%cd /content/gdrive/My Drive/AnimeGAN/ 
```

    Directory  /content/gdrive/My Drive/AnimeGAN/  already exists
    /content/gdrive/My Drive/AnimeGAN



```
# check if data directory exists and is not empty
if os.path.exists(dataset_path) or not is_empty(data_folder) :
  print('Dataset already donwnloaded, leave the codes below commented.')
else:
  print('Download dataset by uncommenting the codes below')

```

    Directory exists
    Directory is not empty
    Dataset already donwnloaded, leave the codes below commented.





## **Working with Kaggle in Colab with the Kaggle API for Downloading Dataset**

> Only follow this steps 1 and 2 once, also you might not need to follow steps 3 and 4 after you have downloaded the dataset once.


> You can use download API token button to download kaggle.json from your kaggle account (kaggle.com/username/account)

> If your token has expried, you'll get a message stating you are unathorized to download the dataset. You can solve this by downlading new token and replacing it in ~/kaggle/kaggle.json. Your token expires if you generate new token or click on expire my token.
[Kaggle API documentation](https://www.kaggle.com/docs/api)





1.   Install Kaggle Library to use Kaggle API
2.   Upload kaggle.json 
3.   list datasets matching a search term: kaggle datasets list -s [KEYWORD]
4.   Download dataset: kaggle datasets download -d [DATASET]
  







```
# !pip install -q kaggle  #install library to communicate with Kaggle API
```


```
# uploaded = files.upload() #upload kaggle.json downloaded form kaggle profile
```


```

# !rm  ~/.kaggle/kaggle.json
# # use if new kaggle.json is downloaded and has to be replaced




# !mkdir -p ~/.kaggle #make a .kaggle folder in root
# !cp kaggle.json ~/.kaggle/ #copy the .kaggle file uploaded to the folder
# !ls ~/.kaggle  #list the files in the .kaggle folder
# !chmod 600 /root/.kaggle/kaggle.json #change the file permisson to read write


```


```
# !kaggle datasets list -s anime-faces #view all the datasets from Kaggle on anime-faces
```


```
# !kaggle datasets download -d soumikrakshit/anime-faces #download anime-faces by soumikrakshit from Kaggle 
#if you get 401-Unauthorized error go to your account in Kaggle, download new kaggle.json, and use the steps above to delete the old one and replace with new
```

## **Unzip Dataset**


```
if is_empty(data_folder) :
  print('The dataset has not been extracted, uncomment the lines below run it and comment it again')
else:
  print('Leave the code below commented')
```

    Directory exists
    The directory is not empty
    Leave the code below commented



```
# !unzip anime-faces.zip #unzip the downloaded dataset; gets unzipped to a folder named data
```


```
#count files in data folder
print(sum([len(files) for r, d, files in os.walk(data_folder)]))

```

    20383



```
#move some data to sampled folder 
if is_empty(sampled_folder):
  move_files(data_folder,sampled_folder,1000)

```

    Directory exists
    The directory is not empty


## **Image to Array**


```
#note if you are using the same dataset, data folder contains a data directory inside it which will 
#throw error so delete the data directory inside data directory
# shutil.rmtree('/content/gdrive/My Drive/AnimeGAN/data/data', ignore_errors=True)

```


```
raw_images = load_array(array_folder,'raw_images.npy',data_folder,load_images) #load images and save to array if not already loaded, else load from saved file
print(f'Dimensions of image array: {raw_images.shape}')
```

    A backup file exists. Loading from the backup file.
    Dimensions of image array: (20383, 64, 64, 3)


## **Display Images**


```
#visualizing image
print(f'Dimensions of image array: {raw_images.shape}')
print(f'Min value of image array: {raw_images.min()}')
print(f'Max value of image array: {raw_images.max()}')
print(f'Mean of image array: {raw_images.mean()}')
print(f'First image value:')

# print(raw_images[0])
print('---------------------------------------------------------------------------------------------------')
print(f'Displaying images\n\n')
display_images(raw_images)

```

    Dimensions of image array: (20383, 64, 64, 3)
    Min value of image array: 0
    Max value of image array: 255
    Mean of image array: 160.7246212248974
    First image value:
    ---------------------------------------------------------------------------------------------------
    Displaying images
    
    
![LOADED IMAGE DATA](https://i.ibb.co/qkRDv9H/Screen-Shot-2020-06-12-at-04-10-01.png)



## **Visualizing data**


```


raw_images_all = np.reshape(raw_images,(raw_images.shape[0]*raw_images.shape[1]*raw_images.shape[2],raw_images.shape[3]))#bring all the pixel value to a single array:shape(total_images*total_pix_in_an_image,3)
# print(raw_images_all[:10])
# print(raw_images_all)
raw_images_r=raw_images_all[:,0]
raw_images_g=raw_images_all[:,1]
raw_images_b=raw_images_all[:,2]

first_image = raw_images[0]#use copy dont assign directly
min = raw_images.min()
max = raw_images.max()
print(f'For dataset, Min: {min},Max:{max}')
print('---------------------------------------------------------------------------------------------------')
print(f'Displaying plot\n\n')
fig = plt.figure(figsize = (16,16))

image_r = first_image.copy()
image_r[:,:,1]=image_r[:,:,2] = 0
image_g = first_image.copy()
image_g[:,:,0]=image_g[:,:,2] = 0
image_b = first_image.copy()
image_b[:,:,0]=image_b[:,:,1] = 0


fig1 = fig.add_subplot(4,4,1)
fig1.imshow(first_image)
fig1.set_title('first image')


fig2 = fig.add_subplot(4,4,2)
fig2.imshow(image_r)
fig2.set_title('first image, red channel')

fig3 = fig.add_subplot(4,4,3)
fig3.imshow(image_g)
fig3.set_title('first image, green channel')

fig4 = fig.add_subplot(4,4,4)
fig4.imshow(image_b)
fig4.set_title('first image, blue channel')

fig5 = fig.add_subplot(4,4,5)
fig5.hist(first_image.flatten(),bins=30)
fig5.set_title('pixel values image')

fig6 = fig.add_subplot(4,4,6)
fig6.hist(first_image[:,:,0].flatten(),bins=30)
fig6.set_title('red values  in first image')

fig7 = fig.add_subplot(4,4,7)
fig7.hist(first_image[:,:,1].flatten(),bins=30)
fig7.set_title('green values  in first image')

fig8 = fig.add_subplot(4,4,8)
fig8.hist(first_image[:,:,2].flatten(),bins=30)
fig8.set_title('blue values  in first image')

fig9 = fig.add_subplot(4,4,9)
fig9.hist(raw_images_all.flatten(),bins=30)
fig9.set_title('pixel values dataset ')

fig10 = fig.add_subplot(4,4,10)
fig10.hist(raw_images_r.flatten(),bins=30)
fig10.set_title('red values  in dataset')

fig11 = fig.add_subplot(4,4,11)
fig11.hist(raw_images_g.flatten(),bins=30)
fig11.set_title('green values  in dataset')

fig12 = fig.add_subplot(4,4,12)
fig12.hist(raw_images_b.flatten(),bins=30)
fig12.set_title('blue values  in dataset')


```

    For dataset, Min: 0,Max:255
    ---------------------------------------------------------------------------------------------------
    Displaying plot
    
    





    Text(0.5, 1.0, 'blue values  in dataset')




![VISUALIZING DATA](https://i.ibb.co/3hf3dC2/Screen-Shot-2020-06-12-at-04-10-26.png)


## **Scaling and Visualizing Data**


```

REAL_IMAGES = load_array(array_folder,'scaled_images.npy',raw_images,normalize_images)#load images and save to array if not already loaded and save, else load from saved file
print('---------------------------------------------------------------------------------------------------')

print(f'Scaled Images')
print(f'Min value after scaling: {REAL_IMAGES.min()}')
print(f'Max value after scaling: {REAL_IMAGES.max()}')
print(f'Mean after scaling: {REAL_IMAGES.mean()}')
print(f'Data type: {REAL_IMAGES.dtype}')

print('---------------------------------------------------------------------------------------------------')

print(f'First Image after Scaling')
# print(REAL_IMAGES[0])
print('---------------------------------------------------------------------------------------------------')

print(f'Raw Image vs Scaled Back Image to make sure they are same:\n')
raw_images_reshaped = np.reshape(raw_images[0],(64*64,3))#bring all the pixel value to a single array:shape(total_images*total_pix_in_an_image,3)
REAL_IMAGES_reshaped = np.reshape(REAL_IMAGES[0],raw_images_reshaped.shape)
print(f'Sample unscaled pixel values image')
print(raw_images_reshaped)
print(f'\nSample scaled pixel values after scaling back')
print(unnormalize_images(REAL_IMAGES_reshaped))
print('---------------------------------------------------------------------------------------------------')


print('Displaying scaled back images')
display_images(unnormalize_images(REAL_IMAGES),start_pos = 0, cols=4, rows=4,fig_size= (10., 10.),grid=222,pad=0)


```

    A backup file exists. Loading from the backup file.
    ---------------------------------------------------------------------------------------------------
    Scaled Images
    Min value after scaling: -1.0
    Max value after scaling: 1.0
    Mean after scaling: 0.26058526450899794
    Data type: float64
    ---------------------------------------------------------------------------------------------------
    First Image after Scaling
    ---------------------------------------------------------------------------------------------------
    Raw Image vs Scaled Back Image to make sure they are same:
    
    Sample unscaled pixel values image
    [[238 177 125]
     [248 220 148]
     [253 228 154]
     ...
     [255 253 251]
     [246 254 248]
     [252 251 252]]
    
    Sample scaled pixel values after scaling back
    [[238 177 125]
     [248 220 148]
     [253 228 154]
     ...
     [255 253 251]
     [246 254 248]
     [252 251 252]]
    ---------------------------------------------------------------------------------------------------
    Displaying scaled back images


![CHECKING IF DATA IS SCALED PROPERLY](https://i.ibb.co/qnYLd7S/Screen-Shot-2020-06-12-at-04-10-41.png)



## **Define required variables**


```
#from dcgan google
# Variables that determine how tensorflow will create batches after data load
BUFFER_SIZE = REAL_IMAGES.shape[0]+1
BATCH_SIZE = 32

# Weight initializers for the Generator network
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)

# Image dimensions, the generator outputs 64x64 size image while the discriminator expects a 64x64
DIM = 64

# Variables needed for the training part
EPOCHS = 30
NOISE_DIM = 100
NUM_OF_EXAMPLES_TO_GENERATE = 16

# Noise Vector to test models
NOISE = tf.random.normal([1,100])

# We will reuse this seed over time (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([NUM_OF_EXAMPLES_TO_GENERATE, NOISE_DIM])
```

## **Casting float64 to float32**
>As gradient functions from Tensorflow require floate32, so we have to type caste float64 to float32


```
#this is needed because the gradient functions from TF require float32 instead of float64
REAL_IMAGES = tf.cast(REAL_IMAGES, 'float32')
```

## **Creating batch and shuffling the data**


```
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(REAL_IMAGES).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

##**Defining Models for our GAN**



> A generative adversarial network (GAN) has two parts:



1.  The generator learns to generate plausible data. The generated instances become negative training examples for the discriminator. 
2.   The discriminator learns to distinguish the generator's fake data from real data. The discriminator penalizes the generator for producing implausible results.





3. When training begins, the generator produces obviously fake data, and the discriminator quickly learns to tell that it's fake


> *All the definations are taken from [Goolge's GAN Page](https://developers.google.com/machine-learning/gan/gan_structure)*





## **GENERATOR MODEL**


> The generator part of a GAN learns to create fake data by incorporating feedback from the discriminator. It learns to make the discriminator classify its output as real.




```
#create a generator model and return it
def build_generator_model(Z=100,WEIGHT_INIT=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2),ALPHA = 0.2,DROP_RATIO =0.3):
  #Sequential helps to create model by staking layers, use an array format or use add to add layers sequentially
  model = tf.keras.Sequential() 

  #Layer 1
  #First layer with 32,768 nodes expecting for an input of vector size 100
  model.add(layers.Dense(8*8*512,use_bias=False,input_shape=(Z,))) 
  # Normalizing activations of the previous layer for each batch
  model.add(layers.BatchNormalization())
  #apply Leaky RELU activation : f(x) = {x if x > 0 : 0.01*x}
  model.add(layers.LeakyReLU()) 
  #reshape the FC layer to (none,8,8,512) where none is batch size
  model.add(layers.Reshape((8,8,512))) 
  # check if the output shape matches the requied shape
  # assert model.output_shape == (None,8,8,512)

  #Layer 2
  #transposed convolution with 256 output_shape, (5,5) filter size, (2,2) strides,
  #the output channel depends on the  outputs shape ie 256 and the shape depends on strids ie (2,2) changes (8,8) tp (16,16)
  #Tansposed Convolution does oppostie of what Convolution layer does
  model.add(layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False,
                kernel_initializer=WEIGHT_INIT))
 
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
  model.add(layers.Dropout(DROP_RATIO)) 
  # assert model.output_shape == (None,16,16,256)

  #Layer 3
  model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False,
                kernel_initializer=WEIGHT_INIT))
 
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(DROP_RATIO)) 
  # assert model.output_shape == (None,32,32,128) 
 

  #Layer 4
  model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False,
                kernel_initializer=WEIGHT_INIT))
 
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(DROP_RATIO)) 
  # assert model.output_shape == (None,64,64,64) 
  #Layer 5, output Layer
  model.add(layers.Dense(3,activation='tanh', use_bias=False,
                kernel_initializer=WEIGHT_INIT))
  # assert model.output_shape == (None,64,64,3)

  return model
```

## **Visualizing the Generator Model**



```
#create instance of generator model
generator = build_generator_model()
#create folder to save summary if not created
create_dir(model_summary_folder)
#create png file of the summary with states and save to the designated folder
tf.keras.utils.plot_model(generator, to_file=model_summary_folder+'generator_model.png', show_shapes=True, show_layer_names=True,rankdir='TB')
# show the summary of generator model
generator.summary()
```

    Directory  /content/gdrive/My Drive/AnimeGAN/models_summary/  already exists
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 32768)             3276800   
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 32768)             131072    
    _________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)    (None, 32768)             0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 8, 8, 512)         0         
    _________________________________________________________________
    conv2d_transpose_3 (Conv2DTr (None, 16, 16, 256)       3276800   
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 16, 16, 256)       1024      
    _________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)    (None, 16, 16, 256)       0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_transpose_4 (Conv2DTr (None, 32, 32, 128)       819200    
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 32, 32, 128)       512       
    _________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)    (None, 32, 32, 128)       0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 32, 32, 128)       0         
    _________________________________________________________________
    conv2d_transpose_5 (Conv2DTr (None, 64, 64, 64)        204800    
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 64, 64, 64)        256       
    _________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)   (None, 64, 64, 64)        0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 64, 64, 64)        0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 64, 64, 3)         192       
    =================================================================
    Total params: 7,710,656
    Trainable params: 7,644,224
    Non-trainable params: 66,432
    _________________________________________________________________

![GENERATOR STATE DIAGRAM](https://i.ibb.co/9GgKXfb/download-1.png)

```
#plot a model graph that can make more complex models easier to understand.
display.Image(filename=model_summary_folder+'generator_model.png')
```





![UNTRAINED GENERATOR OUTPUT FOR NOISE SAMPLE](https://i.ibb.co/z6C5Lrq/Screen-Shot-2020-06-12-at-04-11-07.png)




```
#pass the noise vectro to the instance to generate a fake image
generated_image = generator(NOISE, training=False)
print(f'Shape of generated array: {generated_image.shape}')
print('---------------------------------------------------------------------------------------------------')


print('Displaying Image:')
plt.imshow(generated_image[0])#don't consider the batch
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    Shape of generated array: (1, 64, 64, 3)
    ---------------------------------------------------------------------------------------------------
    Displaying Image:





    <matplotlib.image.AxesImage at 0x7f47ad2e40b8>







## **THE DISCRIMINATOR MODEL**


> The discriminator in a GAN is simply a classifier. It tries to distinguish real data from the data created by the generator. It could use any network architecture appropriate to the type of data it's classifying.




```
#discrimination model
def build_discriminator_model(WEIGHT_INIT=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2),ALPHA = 0.2):

  model = tf.keras.Sequential()

  #Layer 1
  model.add(layers.Conv2D(64,(4,4),strides=(2,2),padding='same',input_shape=[64,64,3],
                          kernel_initializer=WEIGHT_INIT))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  # assert model.output_shape == (None,32,32,64)
  
  #Layer 2
  model.add(layers.Conv2D(128,(4,4),strides=(2,2),padding='same',
                          kernel_initializer=WEIGHT_INIT))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  # assert model.output_shape == (None,16,16,128)

  #Layer 2
  model.add(layers.Conv2D(256,(4,4),strides=(2,2),padding='same',
                          kernel_initializer=WEIGHT_INIT))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  # assert model.output_shape == (None,8,8,256)

  model.add(layers.Flatten())
  model.add(layers.Dense(1,activation="sigmoid"))

  return model

```

**Visualizing Discriminator Model**


```
#create instance of discriminator model
discriminator = build_discriminator_model()
#create png file of the summary with states and save to the designated folder
tf.keras.utils.plot_model(discriminator, to_file=model_summary_folder+'discriminator_model.png', show_shapes=True, show_layer_names=True,rankdir='TB')
discriminator.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_3 (Conv2D)            (None, 32, 32, 64)        3136      
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 32, 32, 64)        256       
    _________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)   (None, 32, 32, 64)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 16, 16, 128)       131200    
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 16, 16, 128)       512       
    _________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)   (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 8, 8, 256)         524544    
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 8, 8, 256)         1024      
    _________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)   (None, 8, 8, 256)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 16384)             0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 16385     
    =================================================================
    Total params: 677,057
    Trainable params: 676,161
    Non-trainable params: 896
    _________________________________________________________________

![DISCRIMINATOR STATE DIAGRAM](https://i.ibb.co/Gsz3kRH/discriminator.png)

```
display.Image(filename=model_summary_folder+'discriminator_model.png')
```








```

decision = discriminator(generated_image)
print(decision)

```

    tf.Tensor([[3.1680302e-31]], shape=(1, 1), dtype=float32)


## **Use Label Smoothing**
[code credit](https://mc.ai/how-to-implement-gan-hacks-to-train-stable-generative-adversarial-networks/)
>It is common to use the class label 1 to represent real images and class label 0 to represent fake images when training the discriminator model.These are called hard labels, as the label values are precise or crisp.

>It is a good practice to use soft labels, such as values slightly more or less than 1.0 or slightly more than 0.0 for real and fake images respectively, where the variation for each image is random.

>This is often referred to as label smoothing and can have a regularizing effect when training the model.


```
# Assign a random integer in range [0.7, 1.0] for positive class and and [0.0, 0.3] for negative class instead of 1/0 labels
def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.3)

def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.5
```

## **Use Noisy Labels**

[code credit](https://mc.ai/how-to-implement-gan-hacks-to-train-stable-generative-adversarial-networks/)

> The labels used when training the discriminator model are always correct.This means that fake images are always labeled with class 0 and real images are always labeled with class 1.It is recommended to introduce some errors to these labels where some fake images are marked as real, and some real images are marked as fake.

> If you are using separate batches to update the discriminator for real and fake images, this may mean randomly adding some fake images to the batch of real images, or randomly adding some real images to the batch of fake images.

> If you are updating the discriminator with a combined batch of real and fake images, then this may involve randomly flipping the labels on some images.



```
# Changing 5% of the real images to 0 for randomly to introduce noise
def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0].value)
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0].value)], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y
```

## **Calculating Loss**


> *We will use Cross-Entropy Loss*

![Corss-entropy Loss](https://i.ibb.co/5FFtmw9/1-rd-Bw0-E-My8-Gu3f-BOB6-GMA.png"
)


> Here 1 labeled for real images and 0 for the fake image. The generator has nothing to do with the last part of the equation as it has nothing to do with real images.

[
Explanation Here](https://machinelearningmastery.com/generative-adversarial-network-loss-functions/)




![alt text](https://i.ibb.co/nBP5GPN/Screen-Shot-2020-06-09-at-18-27-34.png)


>The generator and discriminator losses are different even though they derive from a single formula.





```
#returns an instance of binarycrossentropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
```

## **Converting Values**

* **tf.ones_like()** converts all the values in the provided tesnor to 1

* **tf.zeros_like()** converts all the values in the provided tesnor to 0 



```
#discriminator loss function
def discriminator_loss(real_output, fake_output): 
  #convert all the real output to 1 using tf.ones_like(real_output)
  #apply label smoothing to positive labels using smooth_positive_labels()
  #calulate cross entropy loss 
  real_loss = cross_entropy(smooth_positive_labels(tf.ones_like(real_output)), real_output)
  #convert all the fake output to 0 using tf.zeros_like(fake_output)
  #apply label smoothing to negative labels  using smooth_positive_labels()
  #calulate cross entropy loss 
  fake_loss = cross_entropy(smooth_negative_labels(tf.zeros_like(fake_output)), fake_output)
  #total loss is the sum of real and fake loss 
  #in the formula shown in the image above, the first part before plus accounts for fake loss and the other part for real loss
  total_loss = real_loss + fake_loss
  return total_loss

#generator loss function
def generator_loss(fake_output):
  return cross_entropy(smooth_negative_labels(tf.ones_like(fake_output)), fake_output)


```

**[Optimizaton of loss function](https://medium.com/@tayyipgoren/keras-optimizers-comparison-on-gan-b8b98c3d8645)** using [Adam Optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20an%20optimization%20algorithm,iterative%20based%20in%20training%20data.&text=The%20algorithm%20is%20called%20Adam.)


```
# Using Adam optimizer
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer =  tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
```

## **Save checkpoints**

Use checkpoints to save and restore models, which can be helpful in case a long-running training task is interrupted.


```
#creating checkpoints to save training progress
checkpoint_dir = checkpoint_folder
checkpoint_prefix = os.path.join(checkpoint_folder, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```

## **Define the training loop**
>The training loop begins with the generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.

> The train step function will be called iteratively in the function, which creates noise vectors. No of Vectors: BATCH_SIZE, Length of Vector: NOISE_DIM defined earlier.

>[Gradient Tape](https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/) is a TensorFlow function for Automatic Differentiation.

## What is AD?

* computer has primitive operations available (e.g. addition, multiplication, logarithm)
* so every complicated function can be written as a composition of these primitive functions
* each primitive function has a simple derivative
* AD is a set of techniques using this logic of simple derivatives of composed functions Read this article

**Tensorflow "records" all the operations executed inside the context of a tf.GradientTape onto to a "tape". Using that "tape" and the gradients associated with the recorded operation, TensorFlow computes the gradients of a "recorded" computation using reverse mode differentiation**



```
#from Google's DC GAN

def train_step(images,G_loss_array, D_loss_array):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # the following are the operations recorded onto the "tape"
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      # print(f'Generator Loss: {gen_loss}')
    
      disc_loss = discriminator_loss(real_output, fake_output)
      # print(f'Discriminator Loss: {disc_loss}')
    #append loss to the loss arrays for plotting the loss 
    G_loss_array.append(gen_loss.numpy())
    D_loss_array.append(disc_loss.numpy())
    # the following lines are taking the derivatives and applying gradients using Adam optimizer
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```


```
create_dir(model_folder)
create_dir(discriminator_folder)
create_dir(generator_folder)
create_dir(plot_folder)
```

    Directory  /content/gdrive/My Drive/AnimeGAN/models/  already exists
    Directory  /content/gdrive/My Drive/AnimeGAN/models/discriminator/  already exists
    Directory  /content/gdrive/My Drive/AnimeGAN/models/generator/  already exists
    Directory  /content/gdrive/My Drive/AnimeGAN/models/plot/  already exists



```
def train(dataset, epochs):
    G_loss = []
    D_loss = []

    G_batch_loss =[]
    D_batch_loss =[]
    
    for epoch in range(epochs):

        print(f'Starting epoch {epoch+1}')
        start = time.time()
        batch_count = 1
        print(f'Training batch: {batch_count} ', end = '')
        for image_batch in dataset:
            train_step(image_batch, G_loss, D_loss)
            if (batch_count % 25 == 0):
              print(f' {batch_count} ',end = '')
            if (batch_count % 325 == 0):
              print('')
            batch_count+=1
        print(f'Generator Loss: {G_loss} Discriminator Loss: {D_loss} ')
        plot_loss(G_loss, D_loss, epoch,'Iterations','Loss')
        G_batch_loss.append(np.mean(G_loss))
        D_batch_loss.append(np.mean(D_loss))
        G_loss = []
        D_loss = []
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        if (epoch % 10 == 0):
            checkpoint.save(file_prefix = checkpoint_prefix)
            # display.clear_output(wait=True)
            # generate_and_save_images(generator,
            #                      epoch + 1,
            #                      seed)
            filename = generator_folder+'g_'+str(epoch+1)+'.h5'
            # generator.save(model_folder+str(epoch+1)+'.h5')
            
            #one way of saving model
            discriminator.save(discriminator_folder+'d_'+str(epoch+1)+'.h5',overwrite=True,
                include_optimizer=True)
            #trying diffent way of saving model
            tf.keras.models.save_model(
                generator,
                filename,
                overwrite=True,
                include_optimizer=True,
                save_format=None
            )
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        
            
    # Generate after the final epoch
    print("Final Epoch")
    plot_loss(G_batch_loss, D_batch_loss,None,'Epochs','Loss')
    # display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)
```

## **Generate and save images**


```
#create output folder if not exists
create_dir(output_folder)
```

    Directory  /content/gdrive/My Drive/AnimeGAN/generated_images/  already exists



```
#modified from Google's Code
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(5,5))
  gs1 = gridspec.GridSpec(4, 4)
  gs1.update(wspace=0, hspace=0)
  # fig = plt.figure(figsize=(8,8))

  for i in range(predictions.shape[0]):
      plt.subplot(gs1[i])
      # plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :]+1.)/2.)
      plt.axis('off')
  plt.tight_layout()
  plt.savefig(output_folder+'image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
```

## **Plotting Loss**

Modified from [this code](https://towardsdatascience.com/dcgans-generating-dog-images-with-tensorflow-and-keras-fb51a1071432)


```
def plot_loss(G_losses, D_losses, epoch=None,xlbl='Iterations',ylbl='Loss'):
    plt.figure(figsize=(10,5))
    if not epoch is None:
      plt.title("Generator and Discriminator Loss - EPOCH {}".format(epoch+1))
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend()
    if not epoch is None:
      plt.savefig(plot_folder+'image_at_epoch_{:04d}.png'.format(epoch))
    else:
      plt.savefig(plot_folder+'final_plot.png')
    plt.show()
```


```
# %%time
train(train_dataset, EPOCHS)
```
![Output and losses per epoch](https://i.ibb.co/n6bZMMV/Screen-Shot-2020-06-12-at-04-13-27.png)
![loss per epoch](https://i.ibb.co/CBcCxMJ/Screen-Shot-2020-06-12-at-04-13-43.png)
## **Load Checkpoint**


```
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_folder))
```

## **Save Model**


```
filename = model_folder+'final_generator_model.h5'
tf.keras.models.save_model(
    generator,
    filename,
    overwrite=True,
    include_optimizer=True,
    save_format=None
)


```

## **Generate and Save images to a zip file**


```
create_dir(generated_sample_folder)
```


```
def generate_and_save_images(filename='generated_images.zip',path='./',count=1000):
  z = zipfile.PyZipFile(path+filename, mode='w')
  for k in range(count):
      generated_image = generator(tf.random.normal([1, NOISE_DIM]), training=False)
      f = path+str(k)+'.png'
      img = ((generated_image[0,:,:,:]+1.)/2.).numpy()
      tf.keras.preprocessing.image.save_img(
          f,
          img,
          scale=True
      )
      z.write(f); os.remove(f)
  z.close()
```


```
generate_and_save_images('generated_images.zip',generated_sample_folder,1000)
```

## **View image from any epoch**


```
# Display a single image using the epoch number
def display_image(path,epoch_no):
  return PIL.Image.open(path+'image_at_epoch_{:04d}.png'.format(epoch_no))
```


```
display_image(output_folder,15)
```

![Output from Epoch 15](https://i.ibb.co/Ky05KRb/Screen-Shot-2020-06-12-at-05-25-11.png)




```
display_image(plot_folder,15)
```




![Plot from epoch 15](https://i.ibb.co/NZWTn92/Screen-Shot-2020-06-12-at-05-25-20.png)



## **Use imageio to create an animated gif using the images saved during training.**


```
image_anim_name = 'image_anim.gif'
plot_anim_name = 'plot_anim.gif'
image_anim_path = animation_folder+ image_anim_name
plot_anim_path= animation_folder+ plot_anim_name

```


```
create_dir(animation_folder)
```

    Directory  /content/gdrive/My Drive/AnimeGAN/models/animation/  Created 





    True




```
def create_animation(gif_filename,images_path):
  with imageio.get_writer(gif_filename, mode='I') as writer:
    filenames = glob(images_path+'image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):#check here how it works
      frame = 2*(i**0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
  return filename
  #copied from Google GAN but works from IPYthon verson greater than 6.2
def display_animation(file):
  if IPython.version_info > (6,2,0,''):
    Ipython.display.Image(filename=file)
  else:
    print(f"IPython version mismatch: {IPython.version_info} should be greater thatn (6,2,0,'')")
    print('Cannot display the animation')
```

## **Download animation file**


```
def download_file(path):
  try:
    from google.colab import files
  except ImportError:
    print('Error: Cannot improt google.colab')
    pass
  else:
    files.download(path)
```

## **Creating, Displaying and Saving generated image animation and plot animation**


```
image_animation = create_animation(image_anim_path,output_folder)
display_animation(image_animation)
download_file(image_anim_path)

```


```
plot_animation = create_animation(plot_anim_path,plot_folder)
display_animation(plot_animation)
download_file(plot_anim_path)

```

### 🙏 Thanks for being a patient reader.

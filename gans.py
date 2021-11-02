import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import cv2
import glob
from PIL import Image
from skimage import morphology

tf.random.set_seed(2340)
OUTPUT_CHANNELS = 3
LAMBDA = 100
OUTPUT_CHANNELS = 3
LAMBDA = 100
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
EDGEMAP_DIR = './hed/'
SKETCH_DIR = './hed_processed/'
PAINTING_DIR ='./paintings/'
def post_process_img(img):
  img = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
  img = np.asarray(img, np.float32) / 255
  thinned = morphology.thin(img)
  cleaned = morphology.remove_small_objects(thinned, min_size=128, connectivity=2)
  return cleaned

def post_process_all(in_path, out_path):
  files = glob.iglob(in_path, recursive=True)
  for f in files:
    img = cv2.imread(f, 0)
    img = post_process_img(img)
    image_name = f.split('/')[-1]
    image_name = image_name.split('.')[0]
    cv2.imwrite('{}/{}.jpg'.format(out_path, image_name), img * 255)

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[512,512,3])
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), 
    downsample(128, 4),
    downsample(256, 4),
    downsample(512, 4),
    downsample(512, 4), 
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4),
    upsample(512, 4),
    upsample(256, 4),
    upsample(128, 4),
    upsample(64, 4),
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')
  x = inputs
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')
  x = tf.keras.layers.concatenate([inp, tar])
  down0 = downsample(64, 4, False)(x)
  down1 = downsample(128, 4)(down0)
  down2 = downsample(256, 4)(down1)
  down3 = downsample(512, 4)(down2)
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(1024, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)
  return tf.keras.Model(inputs=[inp, tar], outputs=last)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

def get_edgemap_name(input_file):
  edgemap_dir = tf.constant(EDGEMAP_DIR)
  filename = tf.strings.split(input_file, '/')[-1]
  filename_arr = tf.strings.split(filename, '_')
  filename = tf.strings.join([edgemap_dir,filename_arr[0],'_',filename_arr[1],'_2.jpg'])
  return filename

def grayscale_to_rgb(im):
  return tf.image.grayscale_to_rgb(im)

def get_painting_name(input_file):
  painting_dir = tf.constant(PAINTING_DIR)
  filename = tf.strings.split(input_file, '/')[-1]
  filename_arr = tf.strings.split(filename, '_')
  filename = tf.strings.join([painting_dir,filename_arr[0],'_',filename_arr[1],'.jpg'])
  return filename

def identity(im):
  return im

def load(input_file, get_target_name, post_process_input, post_process_target):
  input = tf.io.read_file(input_file)
  input = tf.image.decode_jpeg(input)
  input = tf.cast(input, tf.float32)
  input = post_process_input(input)
  target_path = get_target_name(input_file)
  target = tf.io.read_file(target_path)
  target = tf.image.decode_jpeg(target)
  target = tf.cast(target, tf.float32)
  target = post_process_target(target)
  return input, target

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT + 60, IMG_WIDTH + 60)

  input_image, real_image = random_crop(input_image, real_image)
  
  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def sketch2paint_load_image_train(image_file):
  input_image, real_image = load(image_file, get_painting_name, grayscale_to_rgb, identity)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

def sketch2paint_load_image_test(image_file):
  input_image, real_image = load(image_file, get_painting_name, grayscale_to_rgb, identity)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
@tf.function
def train_step(input_image, target, epoch,generator,discriminator,generator_optimizer,discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
    
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
    
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))


def fit(train_ds, epochs, test_ds):
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
    for epoch in range(epochs):
        start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch,generator,discriminator,generator_optimizer,discriminator_optimizer)
    print()

    # saving (checkpoint) the model every 10 epochs
    # if (epoch + 1) % 10 == 0:
      # checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  # checkpoint.save(file_prefix = checkpoint_prefix)
  
def get_im_from_path(im_path):
  im = Image.open(im_path)
  im_array = np.asarray(im, dtype=np.float32)
  return im_array

def get_sketch_name_from_painting(input_file):
  sketch_dir = tf.constant(SKETCH_DIR)

  filename = tf.strings.split(input_file, '/')[-1]
  filename = tf.strings.split(filename, '.')[0]
  filename = tf.strings.join([sketch_dir, 
                              filename, 
                              '_3.jpg'])
  return filename.numpy()


def remove_orig_imgs():
  # get all images
  files = glob.iglob('./paintings/*.jpg', recursive=True)
  for f in files:
    # remove grayscale imgs
    im = get_im_from_path(f)
    if(im.shape != (IMG_HEIGHT, IMG_WIDTH, 3)):
      sketch_path = get_sketch_name_from_painting(f)
      
      os.remove(f)
      os.remove(sketch_path)

# Same as post_process_img, except for the cleaning step
def post_process_test_sketch(img):
  img = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
  img = np.asarray(img, np.float32) / 255

  thinned = morphology.thin(img)
  return thinned

def post_process_test_sketches(in_path, out_path):
  files = glob.iglob(in_path, recursive=True)
  for f in files:
    img = cv2.imread(f, 0)
    img = 255 - img
    img = post_process_test_sketch(img)

    image_name = f.split('/')[-1]
    image_name = image_name.split('.')[0]
    cv2.imwrite('{}/{}.jpg'.format(out_path, image_name), img * 255)
    
def generate_painting(model, test_input):
  prediction = model(test_input, training=True)
  fig = plt.figure(figsize=(15,15))
  
  display_list = [test_input[0], prediction[0]]
  title = ['Sketch', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def edgemap_to_painting_predict(path,sketch2paint_generator):
  sketches = glob.iglob(path + '*.jpg', recursive=True)
  for sketch in sketches:
    im = Image.open(sketch)
    im = im.convert('RGB')
    im = np.asarray(im, dtype=np.float32)
    im = tf.constant(im)

    input_image, _ = normalize(im, im)
    input_image = tf.expand_dims(input_image, 0)
    generate_painting(sketch2paint_generator, input_image)
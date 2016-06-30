import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import matplotlib.image as mpimg

print "hello world"

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename


def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders


def load_letter(folder, min_images, image_size, pixel_depth):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)

    num_images = 0
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        try:
            image_data = (ndimage.imread(image_path).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception("unexpected shape: %s" % str(image_data.shape))
            dataset[num_images,:,:] = image_data
            num_images += 1
        except IOError as e:
            print("could not read: ", image_path)

    dataset = dataset[0:num_images,:,:]
    print dataset.shape
    print "mean: ", np.mean(dataset)
    print "std dev: ", np.std(dataset)
    # imgplot = plt.imshow(dataset[11])
    # raw_input("Press Enter to continue...")
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class, 28, 255.0)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def unpickle(pickle_files):
    labels = []
    data = []

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_data = pickle.load(f)
                labels.append(label)
                data.append(letter_data)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return labels, data

def validation_split(labels, data, validation_ratio):
    assert(len(labels) == len(data))

    all_size = len(labels)
    validation_size = all_size // validation_ratio
    train_size = all_size - validation_size

    indices = [x for x in range(all_size)]
    np.random.shuffle(indices)

    validation_


def split_datasets(labels, data, validation_ratio):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class

  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels

num_classes = 10
np.random.seed(133)

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

train_data_filenames = maybe_pickle(train_folders, 45000)
test_data_filenames = maybe_pickle(test_folders, 1800)

train_labels, train_data = unpickle(train_data_filenames)
test_lavels, test_data = unpickle(test_data_filenames)

print train_labels
print len(train_data)
#valid_dataset, valid_labels, train_dataset, train_labels = \
#    merge_datasets(train_data_filenames + test_data_filenames, 0.2)


# train_data = []
# for folder in train_folders:
#     train_data.append(load_letter(folder, 100, 28, 255.0))
#
# test_data = []
# for folder in test_folders:
#     test_data.append(load_letter(folder, 100, 28, 255.0))
#
# assert(len(train_data) == len(test_data))

# print "num letters: ", len(train_data)
# for letter_data in train_data

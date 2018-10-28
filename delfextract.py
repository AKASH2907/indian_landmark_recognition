import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os
import glob
import accumulator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL)

import operator

count = 0
tot = 0
distance_threshold = 0.8
# K nearest neighbors
K = 10

def accumulate(inputs, func):
    itr = iter(inputs)
    prev = next(itr)
    for cur in itr:
        yield prev
        prev = func(prev, cur)

trainpath = './datacluster/ResizedTrain/'
valdir = './datacluster/ResizedVal/'
building_descs = []

m = hub.Module('https://tfhub.dev/google/delf/1')

def get_resized_db_image_paths(destfolder='./datacluster/ResizedTrain'):
    return sorted(list(glob.iglob(os.path.join(destfolder, '*.[Jj][Pp][Ee][Gg]'))))

db_images = get_resized_db_image_paths()
print db_images

image_placeholder = tf.placeholder(
    tf.float32, shape=(None, None, 3), name='input_image')

module_inputs = {
    'image': image_placeholder,
    'score_threshold': 100.0,
    'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
    'max_feature_num': 1000,
}

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

module_outputs = m(module_inputs, as_dict=True)

image_tf = image_input_fn(db_images)

with tf.train.MonitoredSession() as sess:
    results_dict = {}  # Stores the locations and their descriptors for each image
    for image_path in db_images:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        building_descs.append(image_path.split('_')[0].split('/')[-1])
        results_dict[image_path] = sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})
    
locations_agg = np.concatenate([results_dict[img][0] for img in db_images])
descriptors_agg = np.concatenate([results_dict[img][1] for img in db_images])
accumulated_indexes_boundaries = list(accumulate([results_dict[img][0].shape[0] for img in db_images], operator.add))

d_tree = cKDTree(descriptors_agg) # build the KD tree

def compute_locations_and_descriptors(image_path):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    m = hub.Module('https://tfhub.dev/google/delf/1')

    # The module operates on a single image at a time, so define a placeholder to
    # feed an arbitrary image in.
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)

    image_tf = image_input_fn([image_path])

    with tf.train.MonitoredSession() as sess:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        return sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})

def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    '''
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    '''
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index-1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start,accumulated_index_end)

def get_locations_2_use(image_db_index, k_nearest_indices, accumulated_indexes_boundaries):
    '''
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    '''
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(query_image_locations[i])
                locations_2_use_db.append(locations_agg[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)
            
    
for im in os.listdir(valdir):
    
    query_image_locations, query_image_descriptors = compute_locations_and_descriptors(im)
    distances, indices = d_tree.query(
    query_image_descriptors, distance_upper_bound=distance_threshold, k = K, n_jobs=-1)

    # Find the list of unique accumulated/aggregated indexes
    unique_indices = np.array(list(set(indices.flatten())))

    unique_indices.sort()
    if unique_indices[-1] == descriptors_agg.shape[0]:
        unique_indices = unique_indices[:-1]
        
    unique_image_indexes = np.array(
    list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index]) 
              for index in unique_indices])))
    unique_image_indexes
    
    inliers_counts = []
    
    img_1 = mpimg.imread(im)
    for index in unique_image_indexes:
        locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, accumulated_indexes_boundaries)
        # Perform geometric verification using RANSAC.
        _, inliers = ransac(
            (locations_2_use_db, locations_2_use_query), # source and destination coordinates
            AffineTransform,
            min_samples=3,
            residual_threshold=20,
            max_trials=1000)
        # If no inlier is found for a database candidate image, we continue on to the next one.
        if inliers is None or len(inliers) == 0:
            continue
        # the number of inliers as the score for retrieved images.
        inliers_counts.append({"index": index, "inliers": sum(inliers)})
        print('Found inliers for image {} -> {}'.format(index, sum(inliers)))
    
    print inliers_counts
    
    top_match = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)[0]

    index = top_match['index']
    print 'Best guess for this image:', building_descs[index]
    if im.split('_')[0]== building_descs[index]:
        count = count+1
        
    locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, accumulated_indexes_boundaries)
    # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_2_use_db, locations_2_use_query), # source and destination coordinates
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    tot= tot+1
    
print 'Accuracy:', count/tot

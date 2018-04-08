"""

"""

from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;

import os;
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';
import warnings;
warnings.filterwarnings("ignore");

import tensorflow as tf;
import numpy as np;
from datetime import date, time, datetime, timedelta;
from tqdm import tqdm;

# Personal utility library
import util;

try:
    import matplotlib;  # pylint: disable=g-import-not-at-top
    matplotlib.use("TkAgg")  # Need Tk for interactive plots.
    import matplotlib.pyplot as plt; 
    HAS_MATPLOTLIB = True
except ImportError:
  # Plotting requires matplotlib, but the unit test running this code may
  # execute in an environment without it (i.e. matplotlib is not a build
  # dependency). We'd still like to test the TensorFlow-dependent parts of this
  # example.
  HAS_MATPLOTLIB = False

flags = tf.app.flags;
FLAGS = flags.FLAGS;

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.');
flags.DEFINE_string("log_dir", "./logs", " the log dir");
flags.DEFINE_integer("iterations", 5000, "number of iterations");

def main(*args, **kwargs):
    """
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Please install matplotlib to generate a plot from this example.")
 
    ############################################
    # Step 1: Define parameters for the models #
    ############################################
    util.mkdir_p(FLAGS.run_dir);    
    util.logging(FLAGS);    
    tf.logging.info('[Step 1]: Define parameters for the models');
    util.report_param(FLAGS);


    ################################
    # Step 2: Define input dataset #
    ################################
    tf.logging.info('[Step 2]: Define input dataset');
    if FLAGS.data_type == 'random_signal':
        x, y = util.random_signal(length = FLAGS.data_length);

    # Save the data into numpy arra
    np.savez_compressed(os.path.join(FLAGS.run_dir, 'input-data') ,x=x, y=y);
    plt.plot(x, y)
    plt.savefig(os.path.join(FLAGS.run_dir, 'input-timeseries.jpg'));


    for i in tqdm(range(FLAGS.iterations)): # train the model iterations
        pass;
    print("learning rate", FLAGS.learning_rate);

if __name__ == '__main__':
    tf.app.run();



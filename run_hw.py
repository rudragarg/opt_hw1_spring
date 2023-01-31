import anvil.media
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server

import anvil.mpl_util

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

anvil.server.connect("NYNR7O4KBJTB45RIXBDXD555-RDDQF6NOXBMPYLE7")
@anvil.server.callable
def model_run(file):
  print("Called model_run")
  with anvil.media.TempFile(file) as file_name:

    print("File Uploaded")

    input_file = np.genfromtxt(file_name, delimiter=',')  

    input_arr = input_file.reshape((1,28,28,1))
    result = np.argmax(model.predict(input_arr), axis=1)[0]
    

    print("Predicted Number is:", result)
    plt.pcolor( 1-input_file[::-1], cmap = 'gray')
    plt.axis('off')

    print("Returned Image")
    
    
    
    return str(result), anvil.mpl_util.plot_image()

model = tf.keras.models.load_model('final_network')
anvil.server.wait_forever()
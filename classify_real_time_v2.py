from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import cv2
from time import sleep
import numpy as np
from six.moves import urllib
import tensorflow as tf
import time
from gtts import gTTS
import pygame
import os
FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

from threading import Thread
import cv2

# Threaded class for performance improvement
class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
 
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
 
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
 
	def read(self):
		# return the frame most recently read
		return self.frame
 
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

class NodeLookup(object):
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):

    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  create_graph()
  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    vs = WebcamVideoStream(src=0).start()
    sleep(1)
    frame_count=0
    score=0
    start = time.time()
    pygame.mixer.init()
    pred=0
    last=0
    human_string=None

    while True:
          frame = vs.read()
          frame_count+=1
          
          # Only run every 5 frames
          if frame_count%5==0:
              
              # Save the image as the fist layer of inception is a DecodeJpeg
              cv2.imwrite("current_frame.jpg",frame)

              image_data = tf.gfile.FastGFile("./current_frame.jpg", 'rb').read()
              predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})

              predictions = np.squeeze(predictions)
              node_lookup = NodeLookup()
              
              # change flag for more predictions
              top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
              for node_id in top_k:
                human_string_n = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                if score>.5:
                        # Some manual corrections
                        if human_string_n=="stethoscope":human_string_n="Headphones"
                        if human_string_n=="spatula":human_string_n="fork"
                        if human_string_n=="iPod":human_string_n="iPhone"
                        human_string=human_string_n

                        lst=human_string.split()
                        human_string=" ".join(lst[0:2])
                        human_string_filename=str(lst[0])
                
                current= time.time()
                fps=frame_count/(current-start)
                
          # Speech module        
          if last>40 and pygame.mixer.music.get_busy() == False and  human_string==human_string_n:
              pred+=1
              name=human_string_filename+".mp3"
              
              # Only get from google if we dont have it
              if not os.path.isfile(name):
                  tts = gTTS(text="I see a "+human_string, lang='en')
                  tts.save(name)
        
              last=0
              pygame.mixer.music.load(name)
              pygame.mixer.music.play()

          # Show info during some time              
          if last<40 and frame_count>10:
              cv2.putText(frame,human_string, (20,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
              cv2.putText(frame,str(np.round(score,2))+"%", (20,440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

          if frame_count>20:cv2.putText(frame,"fps: "+str(np.round(fps,2)), (460,460), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
              
          cv2.imshow("Frame", frame)
          last+=1


          # if the 'q' key is pressed, stop the loop
          if cv2.waitKey(1) & 0xFF == ord("q"):break

  # cleanup everything
  vs.stop()
  cv2.destroyAllWindows()
  from os import listdir
  from os.path import join

  dir = "/Users/gagol/Desktop/DLHack"
  test=os.listdir(dir)      
  sess.close()
  print("Done")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=1,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

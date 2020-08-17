# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

###
# Derived and modified from https://github.com/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb
###

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from absl import app
from absl import flags
from absl import logging

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

MODELS = {
  'inception_resnet_v2': 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1', 
  'mobilenet_v2': 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1'
}

FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'mobilenet_v2', 'The model to use ' + str(MODELS))
flags.DEFINE_string('imgdir', 'images', 'Directory for input images (default: images)')
flags.DEFINE_string('outdir', 'output', 'Directory for output images (default: output)')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_image_paths(dir_path):
  files = [x for x in os.listdir(dir_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
  print("Total Image Files: {}".format(len(files)))
  return files


def save_image(img_array, save_path):
  print("Saving Image: {}".format(save_path))
  image = Image.fromarray(img_array)
  image.save(save_path, format="JPEG", quality=90)


def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def load_image(image_path, new_height=256, new_width=256, resize=False):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=3)
  if resize:
    img = tf.image.resize(img, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return img


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, dupe, color, font, thickness=4, display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height

  print(display_str_list)
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)

    if dupe:
      left = left + 400
    
    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", 50)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  label_positions = []
  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      box = boxes[i]
      ymin, xmin, ymax, xmax = tuple(box)
      display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
     
      dupe = False
      if label_positions.count(hash(str(box))) > 0:
        print('Duplicate box: {}'.format(box))
        dupe = True
      else: 
        label_positions.append(hash(str(box)))

      draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, dupe, color, font, display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def get_model(model_id):
  MODEL_URL = MODELS.get(model_id)
  print("Configuring Model: {}".format(MODEL_URL))
  signatures = hub.load(MODEL_URL).signatures
  print("Model Signatures: {}".format(signatures))
  return signatures['default']


def main(argv):
  print("TensorFlow Version: {}".format(tf.__version__))
  IMAGES_DIR_PATH = os.path.join(PROJECT_ROOT, FLAGS.imgdir)
  OUTPUT_DIR_PATH = os.path.join(PROJECT_ROOT, FLAGS.outdir)
  #print("GPUs Available: {}".format(tf.config.list_physical_devices('GPU')))

  detector = get_model(FLAGS.model)

  print("Processing images in directory: {}".format(IMAGES_DIR_PATH))
  for file_name in get_image_paths(IMAGES_DIR_PATH):
    image_path = os.path.join(IMAGES_DIR_PATH, file_name)

    print("Loading Image: {}".format(image_path))
    image = load_image(image_path)
    image_tensor = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

    result = detector(image_tensor)
    result = {key: value.numpy() for key,value in result.items()}
    print("Found {} objects.".format(len(result["detection_boxes"])))
    image_with_boxes = draw_boxes(image.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])

    save_path = os.path.join(OUTPUT_DIR_PATH, file_name)
    save_image(image_with_boxes, save_path)
  print("Processed images to directory: {}".format(OUTPUT_DIR_PATH))

if __name__ == '__main__':
  app.run(main)

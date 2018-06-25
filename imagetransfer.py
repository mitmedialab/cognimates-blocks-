from __future__ import print_function
import sys
import asyncio
import websockets
import tensorflow as tf
import librosa
import os
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

BATCH_SIZE = 4
DEVICE = '/gpu:0'


async def scratch_server(websocket, path):
  print("Connected to {!s}".format(websocket.local_address))
  while websocket.open:
      rawData = await websocket.recv()
      process_data(rawData)
      await websocket.send("You dunions")
  print("Connection closed")

def process_data(rawData):
    inputData = rawData.split(",")
    image = inputData[0]
    style = inputData[1]
    in_path = "input/" + image
    out_path = "output/out.jpg"
    checkpoint_dir = style
    print(in_path + " " + out_path + " " + checkpoint_dir)
    main(in_path, out_path, checkpoint_dir)

def startServer():
  start_server = websockets.serve(scratch_server, '127.0.0.1', 3030)
  asyncio.get_event_loop().run_until_complete(start_server)
  asyncio.get_event_loop().run_forever()

def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()


# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, 
            device_t=device_t, batch_size=1)

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def main(in_path, out_path, checkpoint_dir):
    ffwd_to_img(in_path, out_path, checkpoint_dir,
                    device="/gpu:0")

if __name__=='__main__':
    startServer()
import numpy as np
import os
import sys
from six.moves.urllib.request import urlretrieve
import urllib
import socket
import re



urls = ['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02121808', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846']
mypath = '/home/sarthak/PycharmProjects/imagenet/imagenet/dataFiles/'
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

def maybe_download(filename, url,  force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  else:
    print("files already present:", filename)
  return filename

for class_id, url in enumerate(urls):
  maybe_download(mypath + "class"+str(class_id)+".txt",url)

print("Web Page links download Complete ... downloading jpeg images \n")

current_dir = os.listdir(mypath)

for file in current_dir:
  if(re.search(r'.txt', file)):
    errorCount = 0
    downCount = 0
    with open(mypath + file, "r") as f:
      links = f.readlines()

    if not os.path.exists(mypath+"images-"+file.split(".")[0]):
      newdirpath = mypath+"images-"+file.split(".")[0]
      os.makedirs(newdirpath)

      for image_id, link in enumerate(links):
            try:
              urlretrieve(link.strip("\n"), newdirpath+"/"+str(image_id)+".jpeg")
              downCount +=1
              socket.setdefaulttimeout(100)
            except Exception as e:
              print("error downloading")
              errorCount+=1

      print("downloaded:", downCount, "  errorCount:", errorCount)

    else:
      print("class folder already present")

print("images have been downloaded to the path ", mypath)










#!/usr/bin/env python3
import os, sys, urllib.request, re, threading, posixpath, urllib.parse, argparse, random, socket, time, hashlib, pickle, signal, imghdr

#config
output_dir = './bing' #default output dir
pool_sema = threading.BoundedSemaphore(value = 20) #max number of download threads
bingcount = 35 #default bing paging
socket.setdefaulttimeout(2)

in_progress = tried_urls = []
image_md5s = {}
urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
def download(url,output_dir):
	if url in tried_urls:
		return
	pool_sema.acquire() 
	path = urllib.parse.urlsplit(url).path
	filename = posixpath.basename(path).split('?')[0] #Strip GET parameters from filename
	name, ext = os.path.splitext(filename)
	name = name[:36]
	filename = name + ext

	i = 0
	while os.path.exists(os.path.join(output_dir, filename)) or filename in in_progress:
		i += 1
		filename = "%s-%d%s" % (name, i, ext)
	in_progress.append(filename)
	try:
		request=urllib.request.Request(url,None,urlopenheader)
		image=urllib.request.urlopen(request).read()
		if not imghdr.what(None, image):
			print('FAIL: Invalid image, not saving ' + filename)
			return

		md5_key = hashlib.md5(image).hexdigest()
		if md5_key in image_md5s:
			print('FAIL: Image is a duplicate of ' + image_md5s[md5_key] + ', not saving ' + filename)
			return

		image_md5s[md5_key] = filename

		imagefile=open(os.path.join(output_dir, filename),'wb')
		imagefile.write(image)
		imagefile.close()
		print("OK: " + filename)
		tried_urls.append(url)
	except Exception as e:
		print("FAIL: " + filename)
	finally:
		in_progress.remove(filename)
		pool_sema.release()

def fetch_images_from_keyword(keyword,output_dir,max_its):
	current = 1
	last = ''
	while True:
		if current>max_its:
			break
		request_url='https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(keyword) + '&async=content&first=' + str(current) + '&adlt=off'
		request=urllib.request.Request(request_url,None,headers=urlopenheader)
		response=urllib.request.urlopen(request)
		html = response.read().decode('utf8')
		links = re.findall('murl&quot;:&quot;(.*?)&quot;',html)
		try:
			if links[-1] == last:
				return
			last = links[-1]
			current += bingcount
			for link in links:
				t = threading.Thread(target = download,args = (link,output_dir))
				t.start()
		except IndexError:
			print('No search results for "{0}"'.format(keyword))
			return
		time.sleep(0.1)

def backup_history(*args):
	download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'wb')
	pickle.dump(tried_urls,download_history)
	copied_image_md5s = dict(image_md5s)  #We are working with the copy, because length of input variable for pickle must not be changed during dumping
	pickle.dump(copied_image_md5s, download_history)
	download_history.close()
	print('history_dumped')
	if args:
		exit(0)
	
def getImages(query='dog',max_its=200):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	output_dir_origin = output_dir
	signal.signal(signal.SIGINT, backup_history)
	try:
		download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'rb')
		tried_urls=pickle.load(download_history)
		image_md5s=pickle.load(download_history)
		download_history.close()
	except (OSError, IOError):
		tried_urls=[]
	if query:
		fetch_images_from_keyword(query,output_dir,max_its=max_its)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--query', type=str)
	parser.add_argument('--max_pages', type=int)
	args = parser.parse_args()
	getImages(query=args.query,max_its=args.max_pages)
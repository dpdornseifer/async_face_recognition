# Face Recognition Service

This server is just a simple interface for the OpenCV Cascade classifier. 
Every classifying task is executed in a single thread beside the main eventloop 
to keep the application responsive while there are blocking calls as well. 

## Run it on your Mac

### How to run it
1. Create a virtualenv with Python3.4 / Python3.5 (depends on the OpenCV Pyhon extensions Python 3.4 / 3.5).
2. Download the app into the virtualenv directory.
3. Run `pip3 install -r requirements.txt` to install dependencies.
4. Activate virtualenv `source bin/activate`.
5. Make sure that you have the OpenCV3 dependencies installed look at [Install the OpenCV3 Dependencies](## Install the OpenCV3 Dependencies).
6. Set your own host, port and cascade settings in the constructor of the app in line 122.
7. Run the app `python app.py`.
8. Post the image to `host:8080/detectface` as a form parameter with the name `img`. Host and port depends on your settings. 
9. You'll get the coordinates (x, y, width, height) returned in a json response. 
10. If you want to have a look at the picture with the marked objects just do a get on `http://host:8080/` and 
the picture will be returned with the detected objects highlighted. 

### Install the OpenCV3 Dependencies
1. If you are runnding Mac check out [OpenCV install on OSX](http://www.learnopencv.com/install-opencv-3-on-yosemite-osx-10-10-x/).
2. Make sure that you have the `--with-python3` flag set.
3. If you have the library included in your Python `system-site-packages` just add the `--system-site-packages` flag while. 
 creating the virtualenv e.g. `virtualenv -p python3.4 --system-site-packages face_recoginition`. 
 Otherwise just add a `.pth` file in your `/lib/site-packages` folder e.g. `opencv3.pth`. Put in a link to the OpenCV Python
 extensions. If you're running Mac and you have been installing the OpenCV3 package with the Python3 flag, you'll 
 find them in this location `/usr/local/opt/opencv3/lib/python3.4/site-packages`.
4. Make sure that you can run the `import cv2` statement in the Python interpreter without any errors. You can check the 
OpenCV version by using the `cv2.__version__` statement.
5. If you want to use an external haarcascade download one from this [OpenCV GitHub Repository](https://github.com/Itseez/opencv/tree/master/data/haarcascades)
e.g. `curl -O https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml`.


## Run it on Docker
The base image used in the docker file has the OpenCV3 library already installed.
 
1. Clone the Repo and `cd async_race_recognition_service`
2. Set the specific OpenCV cascade file in the docker config (download it from remote or copy it from the local workdir).
3. Build the image: `docker build -t async_race_recognition_service .` 
4. Run the container and attach it to port 8080 of the docker host: `docker run -p 8080:8080 -name async_face -it async_race_recognition_service` 
5. Check the log output: `docker logs -f async_face`


## Example Post
`curl -F "img=@/Users/david/Desktop/people-09.jpg" localhost:8080/detectface`

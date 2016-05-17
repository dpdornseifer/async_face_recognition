FROM harshjv/opencv3

# set proxy for docker environment
#ENV http_proxy http://proxy:8080
#ENV https_proxy https://:8080

# install pip3
RUN apt-get update \
    && apt-get upgrade -qq \
    && apt-get install -y --force-yes \
    python3-pip; \
    apt-get clean

# create the directory and add the application files
RUN mkdir /app
COPY app.py /app/
COPY requirements.txt /app/

# install the required python 3.4 libs
RUN pip3 install -r /app/requirements.txt


# downlaod the haarcascade file
ADD https://raw.githubusercontent.com/Itseez/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml /app/frontalface.xml

# copy or download the haarcascade file
#COPY frontalface.xml /app/frontalface.xml


# set the workdir and start the app
WORKDIR /app
CMD python3.4 app.py

# expose port 8080
EXPOSE 8080

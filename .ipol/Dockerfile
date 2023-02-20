# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
#FROM registry.ipol.im/ipol:v1-py3.9
FROM python:3.11

# install additional debian packages
COPY .ipol/packages.txt packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/* && rm packages.txt
# copy the requirements.txt and install python packages
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt


# copy the code to $bin
ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin
COPY . .
RUN mkdir -p $bin/output && chmod -R 777 $bin



# compile libraries
RUN cd $bin/externas/devernay_1.0 && make clean && make

#Ant Colony Optimization for Estimating Pith Position on Images of Tree Log Ends
#Download code and Instalation
ENV ANT_CENTER_DETECTOR $bin/externas/Ant_colony_for_pith_estimation/
RUN cd $bin/externas && \
    wget https://www.ipol.im/pub/art/2022/338/Ant_colony_for_pith_estimation.zip &&  \
    unzip Ant_colony_for_pith_estimation.zip && \
    mkdir -p $ANT_CENTER_DETECTOR/build && \
    cd $ANT_CENTER_DETECTOR/build && cmake .. && make && \
    chmod -R 777 $ANT_CENTER_DETECTOR

# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH
ENV PATH $ANT_CENTER_DETECTOR:$PATH

# $HOME is writable by the user `ipol`, but
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME

USER ipol

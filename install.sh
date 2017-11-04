 # Results directory
mkdir -p results/ppo

# GYM
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install

# MPI
sudo apt-get install -y mpich
pip3 install mpi4py

# OPENCV DEPS
sudo apt-get install -y qt-sdk
sudo apt-get install -y build-essential checkinstall cmake pkg-config yasm gfortran git
sudo apt-get install -y libjpeg8-dev libjasper-dev libpng12-dev
sudo apt-get install -y libtiff5-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get install -y libxine2-dev libv4l-dev
sudo apt-get install -y libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get install -y libqt4-dev libgtk2.0-dev libtbb-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install -y libvorbis-dev libxvidcore-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y x264 v4l-utils
sudo apt-get install python3-dev python3-pip
sudo -H pip3 install -U pip numpy

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.3.0
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.3.0
cd ..

cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ..

# substitute 4 by output of nproc
make -j4
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig

#find /usr/local/lib/ -type f -name "cv2*.so"
#/usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
#ln -s /usr/local/lib/python3.6/dist-packages/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
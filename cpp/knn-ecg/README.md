## ECG signal classifier based on KNN and ENN methods

Deps:
http://eigen.tuxfamily.org/index.php?title=Main_Page
https://github.com/libigl/libigl

Dependencies can be pulled manually and passed to cmake / eclipse build.
Submodules can be also used to automate this task.

    git submodule update --init 
    
Or via recursive clone 
    
    git clone --recursive https://github.com/wgml/ecg-classification.git

std:
c++11

optimization:
works happily w/ -O3

flags:
-Wall -Wextra -pedantic

## Cmake build

    $ mkdir build
    $ cmake ..
    $ make
    
As a result two binary files will appear.
You can execute test suite with:

    $ ./knn
    $ ./enn

## Eclipse build
Eclipse does not support multiple targets for build. You need to either exclude
knn.cpp or enn.cpp from build with

    Right click on file -> Resource Configurations -> Exclude from build...

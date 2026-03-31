# Installation instructions

These instructions will work best on Unix-like systems

## C++ compilation

Using CMake is the recommended way
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j4
```

## Python side

First, create a virtualenv
```
virtualenv .venv
source .venv/bin/activate
```
then, install the required Python packages
```
pip install -r requirements.txt
```
The cb module is designed for Linux x64.

1. Build the cb module with at least full C++11 support.

```bash
g++ -O3 -g cb.cpp -shared -fPIC -o libcb.so
```

`O3` for speed optimization, `g` for debug info.


2. Inject environment variable

Remember the absoule dir of `libcb.so`.

For example, if `libcb.so` is under `/home/abc/cpython/`. Remember this **absolute path**.

Set two variable with this absolute path:

```bash
ABS_PATH_TO_CB="/home/abc/cpython/"
export LIBC='-L'$ABS_PATH_TO_CB' -lcb -lstdc++'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ABS_PATH_TO_CB
```

`LIBC` is hacked for this python project. `LD_LIBRARY_PATH` is for the dynamic loader.

If your libcb.so is at current directory, you can also use this in bash:

```bash
export LIBC='-L'$PWD' -lcb -lstdc++'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
```

3. Configure Python project
```bash
./configure --with-pydebug
```

4. make
```bash
make -j
```

Enjoy your python with ref leak detector.

It accurately stores the full refcount state, when `sys.gettotalrefcount()` is called, then it will compare the current refcount state with the last one


This fits the pattern of the ref leak tests.

Immortal objects and `PyDictKeysObject` will not be tracked.
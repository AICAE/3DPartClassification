"""
tested on python3 only
usage:
```python
from detect_freecad import append_freecad_mod_path
append_freecad_mod_path()
try:
    import FreeCAD
except ImportError:
    print("freecad is not installed or detectable, exit from this script")
    sys.exit(0)

```
"""

import sys
import subprocess
from shutil import which
import os.path


def is_executable(name):
    """Check whether `name` is on PATH and marked as executable.
    for python3 only, but cross-platform"""

    # from whichcraft import which
    return which(name) is not None


def detect_lib_path(out, libname):
    """parse ldd output and extract the lib, POSIX only
    OSX Dynamic library naming:  lib<libname>.<soversion>.dylib
    """
    # print(type(out))
    output = out.decode("utf8").split("\n")
    for l in output:
        if l.find(libname) >= 0:
            print(l)
            i_start = l.find("=> ") + 3
            i_end = l.find(" (") + 1
            lib_path = l[i_start:i_end]
            return lib_path
    print("dynamic lib file is not found, check the name (without suffix)")


def get_lib_dir(program, libname):
    program_full_path = which(program)
    # print(program_full_path)
    if is_executable("ldd"):
        process = subprocess.Popen(
            ["ldd", program_full_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = process.communicate()
        solib_path = detect_lib_path(out, libname)
        lib_path = os.path.dirname(solib_path)
        if os.path.exists(lib_path):
            return lib_path
        else:
            print("library file " + libname + " is found, but lib dir does not exist")
    else:
        print("ldd is not available, it is not posix OS")


def get_freecad_lib_path():
    ""
    if is_executable("freecad"):  # debian
        FC = "freecad"
    elif is_executable("FreeCAD"):  # fedora
        FC = "FreeCAD"
    elif is_executable("freecad-daily"):
        FC = "freecad-daily"
    else:
        print("FreeCAD is not install")
        return None
    return get_lib_dir(FC, "libFreeCADApp")


def append_freecad_mod_path():
    cmod_path = get_freecad_lib_path()  # c module path
    if cmod_path:
        pymod_path = os.path.join(cmod_path, os.pardir) + os.path.sep + "Mod"
        sys.path.append(cmod_path)
        sys.path.append(pymod_path)


if __name__ == "__main__":
    print(get_freecad_lib_path())

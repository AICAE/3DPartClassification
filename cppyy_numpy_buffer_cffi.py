import cppyy
import ctypes
import numpy as np



cppyy.cppdef("""
    void pass_array(const unsigned char* buf) {
        std::cerr << buf[0] << std::endl;
    }

    void pass_array(const double* buf) {
        std::cerr << buf[0] << std::endl;
    }
    
    enum E{e0, e1, e2};  // as int type in python, but enum constant are avaiable
    enum TE{te0, te1, te2};  // as int type in python
    
""")

# ctypes.c_char_p  is  C-string with NUL ending
c_uchar_p = ctypes.POINTER(ctypes.c_ubyte)

a = np.array('abc')
cppyy.gbl.pass_array(a.ctypes.data_as(c_uchar_p))

# there is no such type of `ctypes.c_double_p`
b=np.ones(2)
c_double_p = ctypes.POINTER(ctypes.c_double)
cppyy.gbl.pass_array(b.ctypes.data_as(c_double_p))

print(dir(cppyy.gbl.E))
print(dir(cppyy.gbl.TE))

te = cppyy.gbl.TE()
print(te + 1, type(te))

# Composing Dask Array with Numba Stencils



########################
if False:
    from cffi import FFI
    ffibuilder = FFI()
    ffibuilder.cdef("my_cffi_module",  # generated python module name
    """
    void pass_array(const double* buf, int count);
    """)

    ffibuilder.set_source("""
        #include <stdio.h>
        void pass_array(const double* buf, int count) {
            printf ("array element count =  %d, with first element =%lf\n", count, buf[0]);
        }
    """)

    ffibuilder.compile(verbose=True)

    import my_cffi_module




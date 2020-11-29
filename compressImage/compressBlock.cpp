// this can be wrapped in pybind11, but split into header and source, compile it

// cv2 C++ API to read image
#include <vector>
#include <bitset>
#include <cassert>
#include <iostream>

// this cause error in cppyy buffer data passing, why?
typedef  unsigned char* ImageType; // numpy array tobytes() C-style layout, 

#define blockSize 8
//static const unsigned int blockSize = 8;
static const unsigned int BitLength = blockSize*blockSize;
uint8_t lightThreshold = 30;
// if normalized, [-1.0, 1.0]
#if blockSize == 8
typedef std::vector<uint64_t> CompressedImageType;
#else
typedef std::vector<uint16_t> CompressedImageType;
#endif


void testBytes(const unsigned char* im, const unsigned int len)
{
    std::cout << "testBytes()" << std::endl;
    for (unsigned int xi = 0; xi < len; xi++)
        std::cout << int(im[xi])<< ", ";
    std::cout << std::endl;
}


void testArray(void * a, const unsigned int len)
{
    const unsigned char* im = (unsigned char*)a;
    std::cout << "testArray()" << std::endl;
    for (unsigned int xi = 0; xi < len; xi++)
        std::cout << int(im[xi])<< ", ";
    std::cout << std::endl;
}

// padding outside before call this function, void* is converted into LowLevelView
void* compressBlock(const unsigned char* im, const std::vector<int> shape)
{
    size_t width = shape[0];  // im.Width();
    size_t height = shape[1]; // im.Height();
    assert(width > 0 && height > 0);
    assert(width % blockSize == 0 && height % blockSize == 0);
    size_t xBlockCount = width / blockSize;
    size_t yBlockCount = height / blockSize;
    //std::cout << "cim shape: " << yBlockCount  << ", " << xBlockCount << "\n";

    // who has the response to delete memory?
    CompressedImageType* m = new CompressedImageType();
    CompressedImageType& cim = (*m);
    cim.resize(xBlockCount*yBlockCount);

    // size_t BitLength = blockSize*blockSize;

    for (size_t y = 0; y < yBlockCount; y++) // Y coordinate
    {
        for (size_t x = 0; x < xBlockCount; x++)
        {
            std::bitset<BitLength> myBitset;
            for (unsigned int yi = 0; yi < blockSize; yi++)
            {
                for (unsigned int xi = 0; xi < blockSize; xi++)
                {
                    // assuming C array layout, Endianness?
                    size_t xi_ = x * blockSize + xi;
                    size_t yi_ = y * blockSize + yi;
                    // value(x, y) is faster, does not check color type
                    //bool white = (im.PixelColor(xi_, yi_)).GetRGB().Green() > 0.1;
                    bool white = im[yi_ * width + xi_] > lightThreshold;
                    if (white) // binary image conversion
                        myBitset[yi * blockSize + xi] = true;
                    else
                        myBitset[yi * blockSize + xi] = false;
                    //if(y==0 && x==0)  // << yi_ * width + xi_<< ":"
                    //    std::cout << int(im[yi_ * width + xi_])<< ", ";
                }
            }
            unsigned int pos = blockSize * y + x;
            #if blockSize == 8
            cim[pos] = myBitset.to_ullong(); // only for uint64_t?
            #else
            cim[pos] = static_cast<uint16>(myBitset.to_ullong()); 
            #endif
        }
    }
    return cim.data();
}

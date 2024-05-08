#ifndef CALLS_CUH
#define CALLS_CUH
#include <algorithm>
namespace GPUInterface {
    enum DrawType {
        Triangle,
        Line,
        Point
    };
    struct DrawCall {
        int drawtype;
        int indexcount;
        int material;
        int* indexbuffer;
        DrawCall(int drawtype, int indexcount, int material, int* indexbuffer) {
            this->drawtype = drawtype;
            this->indexcount = indexcount;
            this->material = material;
            this->indexbuffer = new int[indexcount];
            std::memcpy(this->indexbuffer, indexbuffer, sizeof(int) * indexcount);
        }
        DrawCall(DrawCall& copy) {
            drawtype = copy.drawtype;
            indexcount = copy.indexcount;
            material = copy.material;
            std::memcpy(indexbuffer, copy.indexbuffer, sizeof(int) * indexcount);
        }
        void Dump(int* &dumped) {
            dumped = new int[3 + indexcount]();
            dumped[0] = drawtype;
            dumped[1] = indexcount;
            dumped[2] = material;
            std::memcpy(&(dumped[3]), &(indexbuffer[0]), sizeof(int) * indexcount);
        }
    };
}
#endif
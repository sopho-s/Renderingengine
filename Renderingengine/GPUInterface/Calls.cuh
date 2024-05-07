#ifndef CALLS_CUH
#define CALLS_CUH
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
        DrawCall(DrawCall& copy) {
            drawtype = copy.drawtype;
            indexcount = copy.indexcount;
            material = copy.material;
            std::copy(copy.indexbuffer, copy.indexbuffer + copy.indexcount, indexbuffer);
        }
    };
    class Caller {
    private:
        DrawCall stack[500];
        DrawCall* __shared__ sharedstack[500];
        int startpointer, endpointer;
    public:
        Caller();
        Caller operator<<(int* data);
    };
}
#endif
namespace GPUInterface {
    class Caller {
    private:
        int* stack
        int stack[500];
        int startpointer, endpointer;
    public:
        Caller();
        Caller operator<<(int* data);
    };
}
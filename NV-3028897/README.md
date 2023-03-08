# NV [3028897](https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun): CUDA: "missing return statement at end of non-void function" in constexpr if function

When I compile the following test code, I get this warning:

```console
test.cu(49): warning: missing return statement at end of non-void function "AllocateSize<T,D>(size_t) noexcept [with T=int, D=Device::GPU]"
          detected during instantiation of "Pointer<T, D> AllocateSize<T,D>(size_t) noexcept [with T=int, D=Device::GPU]" 
(61): here
Should I be concerned and it and is this expected? What can I do to make it go away? This seems weird since cuda does support C++17. Thanks in advance!
```

Compiled with: `nvcc -std=c++17 test.cu -o test`.

The test code (test.cu):

```cuda
enum class Device { CPU, GPU }; // Device

template <typename T, Device D>
class Pointer {
private:
    T* m_raw = nullptr;
    
public:
    __host__ __device__ inline Pointer(T* const p)              noexcept { this->SetPointer(p); }

    __host__ __device__ inline void SetPointer(const Pointer<T, D>& o) noexcept { this->m_raw = o.m_raw; }

    template <typename U>
    __host__ __device__ inline Pointer<U, D> AsPointerTo() const noexcept {
        return Pointer<U, D>(reinterpret_cast<U*>(this->m_raw));
    }

    __host__ __device__ inline operator T*& () noexcept { return this->m_raw; }
}; // Pointer<T, D>

template <typename T>
using CPU_Ptr = Pointer<T, Device::CPU>;

template <typename T>
using GPU_Ptr = Pointer<T, Device::GPU>;

template <typename T, Device D>
__host__ inline Pointer<T, D> AllocateSize(const size_t size) noexcept {
    if constexpr (D == Device::CPU) {
        return CPU_Ptr<T>(reinterpret_cast<T*>(std::malloc(size)));
    } else {
        T* p;
        cudaMalloc(reinterpret_cast<void**>(&p), size);
        return GPU_Ptr<T>(p);
    }
}

template <typename T, Device D>
__host__ inline void Free(const Pointer<T, D>& p) noexcept {
    if constexpr (D == Device::CPU) {
        std::free(p);
    } else {
        cudaFree(p.template AsPointerTo<void>());
    }
}

int main() { Free(AllocateSize<int, Device::GPU>(1024)); }
```

- CUDA release 11.1
- Ubuntu Based Linux Distro


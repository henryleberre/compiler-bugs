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

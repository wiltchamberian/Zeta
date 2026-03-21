#pragma once 

#include <iomanip>
#include <vector>
#include <memory>
#include <ranges>
#include <cassert>
#include <initializer_list>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <variant>
#include <functional>

namespace zeta {

    using Index = int;

    // a purely wrapper of vector
    struct Shape {
        std::vector<Index> dims;

        Shape() = default;
        template<typename Iter>
        Shape(Iter first, Iter last) : dims(first, last) {}
        Shape(std::initializer_list<Index> il) : dims(il) {}
        Shape(int dim, const Shape& sp) {
            dims.resize(1 + sp.size());
            dims[0] = dim;
            for (int i = 0; i < sp.size(); ++i) {
                dims[i + 1] = sp.dims[i];
            }
        }

        std::vector<Index>::iterator begin() {
            return dims.begin();
        }

        std::vector<Index>::const_iterator begin() const {
            return dims.begin();
        }

        std::vector<Index>::iterator end() {
            return dims.end();
        }

        std::vector<Index>::const_iterator end() const {
            return dims.end();
        }

        void resize(const size_t siz) {
            dims.resize(siz);
        }

        Index& back() {
            return dims.back();
        }

        const Index& back() const {
            return dims.back();
        }

        Index& front() {
            return dims.front();
        }

        const Index& front() const {
            return dims.front();
        }

        Shape RemoveFirstDim() const {
            Shape result(this->dims.begin() + 1, this->dims.end());
            return result;
        }

        void clear() {
            dims.clear();
        }

        // 支持正向索引和负索引
        Index operator[](int idx) const {
            size_t n = dims.size();
            if (idx < 0) {
                idx += n;
            }
            assert(idx >= 0 && idx < static_cast<int>(n));
            return dims[idx];
        }

        Index& operator[](int idx) {
            size_t n = dims.size();
            if (idx < 0) {
                idx += n;
            }
            assert(idx >= 0 && idx < static_cast<int>(n));
            return dims[idx];
        }

        bool operator == (const Shape& other) const {
            return dims == other.dims;
        }

        bool empty() const {
            return dims.empty();
        }

        size_t size() const { return dims.size(); }

        void push_back(Index x) { dims.push_back(x); }
    };

    // ============================
    // Slice 类
    // ============================
    struct Slice {
        Index start = 0;
        Index end = 0;   // 0 表示默认到维度末尾
        Index step = 1;

        Slice() = default;                  // {}
        Slice(Index s, Index e, Index st = 1)
            : start(s), end(e), step(st) {
        }

        Slice(std::initializer_list<Index> il) {
            auto it = il.begin();
            if (il.size() == 1) {
                start = it[0];
                end = 0;
                step = 1;
            }
            else if (il.size() == 2) {
                start = it[0];
                end = it[1];
                step = 1;
            }
            else if (il.size() == 3) {
                start = it[0];
                end = it[1];
                step = it[2];
            }
            else {
                throw std::invalid_argument("Slice initializer_list must have 1~3 elements");
            }
        }
    };

    // ============================
    // Tensor 类
    // ============================
    template<typename T = float>
    class TensorT {
    public:
        using ElementType = T;
        static const int end = 0;
        // -------------------
        // 构造
        // -------------------
        TensorT() :offset(0) {} // 默认构造
        TensorT(const Shape& dims) : shape(dims) { initData(); }
        template<typename... Dims>
        TensorT(Dims... dims) : shape{ static_cast<Index>(dims)... } {
            initData();
        }
        // --------------------
        // 拷贝构造
        // --------------------
        TensorT(const TensorT<T>& other)
            : shape(other.shape), stride(other.stride),
            offset(other.offset), data_(other.data_) // 共享数据
        {
        }

        // --------------------
        // 拷贝赋值
        // --------------------
        TensorT<T>& operator=(const TensorT<T>& other) {
            if (this != &other) {
                shape = other.shape;
                stride = other.stride;
                offset = other.offset;
                data_ = other.data_; // 共享数据
            }
            return *this;
        }

        // --------------------
        // 移动构造
        // --------------------
        TensorT(TensorT<T>&& other) noexcept
            : shape(std::move(other.shape)),
            stride(std::move(other.stride)),
            offset(other.offset),
            data_(std::move(other.data_))
        {
            other.offset = 0;
        }

        // --------------------
        // 移动赋值
        // --------------------
        TensorT<T>& operator=(TensorT<T>&& other) noexcept {
            if (this != &other) {
                shape = std::move(other.shape);
                stride = std::move(other.stride);
                offset = other.offset;
                data_ = std::move(other.data_);

                other.offset = 0;
            }
            return *this;
        }

        TensorT<T> Clone() const {
            TensorT<T> ten(this->shape);
            ten.copy(*this); //FIX ME , a quick hack
            return ten;
        }

        bool isEmpty() const {
            return (numel() == 0) || (data_ == nullptr);
        }

        void setData(std::initializer_list<float> list) {
            assert(numel() == list.size());
            assert(is_continuous());
            std::copy(list.begin(), list.end(), data_->begin());
        }

        void setData(std::initializer_list<double> list) {
            assert(numel() == list.size());
            assert(is_continuous());
            std::copy(list.begin(), list.end(), data_->begin());
        }

        void setData(const std::vector<float>& list) {
            assert(numel() == list.size());
            assert(is_continuous());
            std::copy(list.begin(), list.end(), data_->begin());
        }

        void setData(const std::vector<double>& list) {
            assert(numel() == list.size());
            assert(is_continuous());
            std::copy(list.begin(), list.end(), data_->begin());
        }

        template<typename... Dims>
        void zeros(Dims... dims) {
            shape = Shape{ static_cast<Index>(dims)... };
            initData();
        }

        void constants(T value) {
            if (data_) {
                std::fill(data_->begin(), data_->end(), value);
            }
        }

        void zeros(const Shape& shape) {
            this->shape = shape;
            initData();
        }

        // ------------- 初始化底层数据 -------------
        void initData() {
            // 计算元素总数
            Index total = 1;
            for (Index s : shape) {
                if (s <= 0) assert(false);
                total *= s;
            }

            // 分配 shared_ptr vector
            data_ = std::make_shared<std::vector<T>>(total, 0);

            // 初始化 stride (C-contiguous)
            stride.resize(shape.size());
            if (!shape.empty()) {
                stride[shape.size() - 1] = 1;
                for (int i = int(shape.size()) - 2; i >= 0; --i)
                    stride[i] = shape[i + 1] * stride[i + 1];
            }

            // 偏移清零
            offset = 0;
        }

        // -------------------
        // 元信息
        // -------------------
        Index rank() const { return shape.size(); }
        size_t ElementCount() const { return data_ ? data_->size() : 0; }
        size_t size() const {
            return numel();
        }
        size_t numel() const {
            if (shape.empty()) {
                return 0;
            }
            size_t z = 1;
            for (auto& s : shape) {
                z *= s;
            }
            return z;
        }

        T* start() {
            return data_ ? ((*data_).data() + offset) : nullptr;
        }

        const T* start() const {
            return data_ ? ((*data_).data() + offset) : nullptr;
        }

        T* data() {
            return data_ ? (*data_).data() : nullptr;
        }

        const T* data() const {
            return data_ ? (*data_).data() : nullptr;
        }

        bool is_continuous() const {
            //if (offset != 0) return false;
            if (shape.empty()) return true;

            size_t expected = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                if (stride[i] != expected)
                    return false;
                expected *= shape[i];
            }
            return true;
        }

        void copy(const TensorT<T>& tensor) {
            //fast copy
            if (is_continuous() && tensor.is_continuous()) {
                int siz = numel();
                if (siz == tensor.numel()) {
                    std::copy(tensor.start(), tensor.start() + siz, this->start());
                    return;
                }
                else {
                    assert(false);
                }
            }
            assert(false);
        }

        TensorT<T> contiguous() const {
            if (is_continuous()) {
                return *this;
            }
            TensorT<T> result(this->shape);

            size_t R = shape.size();
            size_t total = numel();
            std::vector<Index> idx(R, 0); // 多维索引

            for (size_t count = 0; count < total; ++count) {
                // 计算 flat offset
                Index pos_this = offset;
                Index pos_result = result.offset;
                for (size_t i = 0; i < R; ++i) {
                    pos_this += idx[i] * stride[i];
                    pos_result += idx[i] * result.stride[i];
                }

                (*result.data_)[pos_result] = (*data_)[pos_this];

                // 多维索引进位（odometer）
                for (int d = R - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }

            return result;
        }

        // -------------------
        // 返回第一个维度最后一个子张量
        // -------------------
        TensorT<T> back() const {
            assert(!shape.empty());
            assert(shape[0] > 0);

            TensorT<T> res;
            res.data_ = data_;                 // 共享数据
            res.offset = offset + (shape[0] - 1) * stride[0];
            res.shape = Shape(shape.begin() + 1, shape.end());
            res.stride = Shape(stride.begin() + 1, stride.end());
            return res;
        }
        // -------------------
        // 一维访问
        // -------------------
        TensorT<T> operator[](Index i) {
            assert(i >= 0 && i < this->shape[0]);
            TensorT<T> result = *this;

            // scalar case: rank == 1 → 返回 0-rank tensor
            if (this->rank() == 1) {
                result.shape.clear();
                result.stride.clear();
                result.offset = this->offset + i * this->stride[0];
                return result;
            }


            result.shape = Shape(this->shape.dims.begin() + 1, this->shape.dims.end());
            result.offset = this->offset + i * this->stride[0];
            result.stride = Shape(this->stride.dims.begin() + 1, this->stride.dims.end());
            return result;
        }

        T& operator()(size_t a) {
            assert(1 == shape.size());
            int pos = offset + a * stride[0];
            return (*data_)[pos];
        }

        const T& operator()(size_t a) const {
            assert(1 == shape.size());
            int pos = offset + a * stride[0];
            return (*data_)[pos];
        }

        // ------------------------

        T& operator()(size_t a, size_t b) {
            assert(2 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1];
            return (*data_)[pos];
        }

        const T& operator()(size_t a, size_t b) const {
            assert(2 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1];
            return (*data_)[pos];
        }

        // ------------------------

        T& operator()(size_t a, size_t b, size_t c) {
            assert(3 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1] + c * stride[2];
            return (*data_)[pos];
        }

        const T& operator()(size_t a, size_t b, size_t c) const {
            assert(3 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1] + c * stride[2];
            return (*data_)[pos];
        }

        // ------------------------

        T& operator()(size_t a, size_t b, size_t c, size_t d) {
            assert(4 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1]
                + c * stride[2] + d * stride[3];
            return (*data_)[pos];
        }

        const T& operator()(size_t a, size_t b, size_t c, size_t d) const {
            assert(4 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1]
                + c * stride[2] + d * stride[3];
            return (*data_)[pos];
        }

        // ------------------------

        T& operator()(size_t a, size_t b, size_t c, size_t d, size_t e) {
            assert(5 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1]
                + c * stride[2] + d * stride[3] + e * stride[4];
            return (*data_)[pos];
        }

        const T& operator()(size_t a, size_t b, size_t c, size_t d, size_t e) const {
            assert(5 == shape.size());
            int pos = offset + a * stride[0] + b * stride[1]
                + c * stride[2] + d * stride[3] + e * stride[4];
            return (*data_)[pos];
        }

        // ---------------------------
        // 多维索引 operator() 返回标量
        // ---------------------------
        template<typename... Args>
        T& operator()(Args... args) {
            static_assert(sizeof...(Args) > 0, "Must provide at least one index");
            static_assert(sizeof...(Args) == sizeof...(Args), "Dim mismatch"); // 可加 runtime check

            Index pos = offset;
            Index idxs[] = { static_cast<Index>(args)... };
            assert(sizeof...(Args) == shape.size());

            for (size_t i = 0; i < shape.size(); ++i) {
                assert(idxs[i] < shape[i]);
                pos += idxs[i] * stride[i];
            }

            return (*data_)[pos];
        }

        // const 版本
        template<typename... Args>
        const T& operator()(Args... args) const {
            Index pos = offset;
            Index idxs[] = { static_cast<Index>(args)... };
            assert(sizeof...(Args) == shape.size());

            for (size_t i = 0; i < shape.size(); ++i) {
                assert(idxs[i] < shape[i]);
                pos += idxs[i] * stride[i];
            }

            return (*data_)[pos];
        }


        template<typename T>
        TensorT<T> operator+(const TensorT<T>& other) const {
            assert(shape == other.shape); // shape 必须一致
            TensorT<T> result(shape);

            size_t R = shape.size();
            size_t total = numel();
            std::vector<Index> idx(R, 0); // 多维索引

            for (size_t count = 0; count < total; ++count) {
                // 计算 flat offset
                Index pos_this = offset;
                Index pos_other = other.offset;
                Index pos_result = result.offset;
                for (size_t i = 0; i < R; ++i) {
                    pos_this += idx[i] * stride[i];
                    pos_other += idx[i] * other.stride[i];
                    pos_result += idx[i] * result.stride[i];
                }

                // 元素加法
                (*result.data_)[pos_result] = (*data_)[pos_this] + (*other.data_)[pos_other];

                // 多维索引进位（odometer）
                for (int d = R - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }

            return result;
        }

        template<typename T>
        TensorT<T> operator-(const TensorT<T>& other) const {
            assert(shape == other.shape); // shape 必须一致
            TensorT<T> result(shape);

            size_t R = shape.size();
            size_t total = numel();
            std::vector<Index> idx(R, 0); // 多维索引

            for (size_t count = 0; count < total; ++count) {
                // 计算 flat offset
                Index pos_this = offset;
                Index pos_other = other.offset;
                Index pos_result = result.offset;
                for (size_t i = 0; i < R; ++i) {
                    pos_this += idx[i] * stride[i];
                    pos_other += idx[i] * other.stride[i];
                    pos_result += idx[i] * result.stride[i];
                }

                // 元素减法
                (*result.data_)[pos_result] = (*data_)[pos_this] - (*other.data_)[pos_other];

                // 多维索引进位
                for (int d = R - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }

            return result;
        }



        // -------------------
        // Hadamard 积
        // -------------------
        TensorT<T> operator%(const TensorT<T>& other) const {
            assert(shape == other.shape);
            TensorT<T> result(shape);

            size_t R = shape.size();
            size_t total = numel();
            std::vector<Index> idx(R, 0);

            for (size_t count = 0; count < total; ++count) {
                // 计算 flat offset
                Index pos_this = offset;
                Index pos_other = other.offset;
                Index pos_result = result.offset;
                for (size_t i = 0; i < R; ++i) {
                    pos_this += idx[i] * stride[i];
                    pos_other += idx[i] * other.stride[i];
                    pos_result += idx[i] * result.stride[i];
                }

                // Hadamard
                (*result.data_)[pos_result] = (*data_)[pos_this] * (*other.data_)[pos_other];

                // 进位更新 idx
                for (int d = R - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }

            return result;
        }

        template<typename T>
        TensorT<T> matmul(const TensorT<T>& B) const {
            const TensorT<T>& A = *this;

            assert(A.rank() >= 1 && B.rank() >= 1);

            // -----------------------------
            // 1️⃣ 确定矩阵维度
            // -----------------------------
            size_t A_rank = A.rank();
            size_t B_rank = B.rank();

            Index A_last = A.shape.back();          // A 的最后一维
            Index B_first = B.shape[0];             // B 的第一维

            assert(A_last == B_first);              // 矩阵乘法必须匹配

            // batch shape = A_rank-1 前缀 + B_rank-1 后缀
            Shape out_shape;

            // A 前缀 batch (除了最后一维)
            for (size_t i = 0; i < A_rank - 1; ++i)
                out_shape.push_back(A.shape[i]);

            // B 后缀 (除了第一维)
            for (size_t i = 1; i < B_rank; ++i)
                out_shape.push_back(B.shape[i]);

            TensorT<T> result(out_shape);

            // -----------------------------
            // 2️⃣ batch odometer
            // -----------------------------
            size_t total = result.numel();
            std::vector<Index> idx(result.rank(), 0);

            for (size_t count = 0; count < total; ++count) {
                // -----------------------------
                // 3️⃣ 计算 flat offsets
                // -----------------------------
                Index posA = A.offset;
                Index posB = B.offset;
                Index posR = result.offset;

                // A 前缀 batch
                size_t batch_len = A_rank - 1;
                for (size_t i = 0; i < batch_len; ++i) {
                    Index dim = idx[i];
                    posA += dim * A.stride[i];
                }

                // B 前缀 batch (如果 B rank>1)
                for (size_t i = 1; i < B_rank; ++i) {
                    Index dim = idx[batch_len + i - 1];
                    posB += dim * B.stride[i];
                }

                for (size_t i = 0; i < idx.size(); ++i) {
                    posR += idx[i] * result.stride[i];
                }

                // -----------------------------
                // 4️⃣ 矩阵乘法核心
                // A shape: (..., K)
                // B shape: (K, ...)
                // -----------------------------
                Index K = A_last;
                for (size_t i = 0; i < K; ++i) {
                    Index a_idx = posA + i * A.stride[A_rank - 1];
                    Index b_idx = posB + i * B.stride[0];

                    // A_last_dim = 1 ? ... 这里我们做简单累加到结果
                    // 由于 output stride 对应 result stride，我们可以直接累加
                    (*result.data_)[posR] += (*A.data_)[a_idx] * (*B.data_)[b_idx];
                }

                // -----------------------------
                // 5️⃣ 多维索引进位
                // -----------------------------
                for (int d = int(idx.size()) - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < result.shape[d]) break;
                    idx[d] = 0;
                }
            }

            return result;
        }



        //slice
        template<typename... Slices>
        TensorT<T> slice(Slices... slices) const {
            constexpr int N = sizeof...(Slices);
            assert(N <= rank());

            //Slice tmp[] = { Slice(std::forward<Args>(args))... };
            Slice tmp[] = { slices... };
            TensorT<T> result = *this;

            for (int i = 0; i < N; ++i) {
                const Slice& s = tmp[i];

                Index dim = this->shape[i];
                Index start = s.start;
                Index end = (s.end == 0) ? dim : s.end;
                Index step = s.step;

                assert(step > 0);
                assert(start >= 0 && start < end && end <= dim);

                result.offset += start * this->stride[i];
                result.stride[i] = step * this->stride[i];

                result.shape[i] = (end - start + step - 1) / step;
            }

            return result;
        }

        // ------ 
        // permute
        // input must be permutation of {0,1,2,...,rank()-1};
        //
        TensorT<T> permute(const Shape& shape) const {
            assert(shape.size() == this->rank());
            TensorT<T> newTensor = *this;
            newTensor.offset = this->offset;
            int count = this->rank();
            for (int i = 0; i < count; ++i) {
                newTensor.shape[i] = this->shape[shape[i]];
                newTensor.stride[i] = this->stride[shape[i]];
            }

            return newTensor;
        }

        // -------------------
        // view / reshape
        // -------------------
        TensorT<T> view(const Shape& newShape) const {
            size_t total = 1;
            for (Index s : newShape) total *= s;
            assert(total == ElementCount()); // reshape 元素总数必须一致

            TensorT<T> v;
            v.data_ = data_;
            v.offset = offset;
            v.shape = newShape;

            v.stride.resize(newShape.size());
            if (!newShape.empty()) {
                v.stride[newShape.size() - 1] = 1;
                for (int i = int(newShape.size()) - 2; i >= 0; --i)
                    v.stride[i] = newShape[i + 1] * v.stride[i + 1];
            }
            return v;
        }

        template<typename... Dims>
        TensorT<T> reshape(Dims... dims) const {
            Shape newShape{ static_cast<Index>(dims)... };
            return view(newShape); // 直接调用已有的 view 函数
        }

        // operator* 普通张量内积
        TensorT<T> operator*(const TensorT<T>& other) const {
            return this->matmul(other);
        }

        void save(std::fstream& fs) {
            int rk = rank();
        }

        void print(std::string prefix) const {
            size_t R = rank();
            size_t total = numel();
            std::vector<Index> idx(R, 0); // 多维索引

            for (size_t count = 0; count < total; ++count) {
                // 计算 flat offset
                Index pos_this = offset;
                std::cout << prefix;
                for (size_t i = 0; i < R; ++i) {
                    pos_this += idx[i] * stride[i];
                    std::cout << idx[i] << ((i == R - 1) ? "" : ",");
                }
                std::cout << "=" << std::fixed << std::setprecision(4) << (*data_)[pos_this] << " ";

                // 多维索引进位
                for (int d = R - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }
            std::cout << std::endl;

        }

        void print_torch_style(std::string prefix = "") const {
            size_t R = rank();
            std::vector<Index> idx(R, 0); // 多维索引

            std::function<void(size_t, size_t, std::vector<Index>&, int)> print_dim;
            print_dim = [&](size_t dim, size_t offset_curr, std::vector<Index>& idx, int indent) {
                std::string indent_str(indent * 2, ' ');
                std::cout << indent_str << "[";
                size_t N = shape[dim];
                for (size_t i = 0; i < N; ++i) {
                    idx[dim] = i;
                    size_t pos = offset_curr + i * stride[dim];
                    if (dim + 1 < R) {
                        std::cout << "\n";
                        print_dim(dim + 1, pos, idx, indent + 1);
                        if (i != N - 1) std::cout << ",";
                    }
                    else {
                        double val = (*data_)[pos];
                        if ((val != 0.0) && (std::abs(val) < 1e-4 || std::abs(val) > 1e4)) {
                            // 科学计数法
                            std::cout << std::scientific << std::setprecision(5) << val;
                        }
                        else {
                            // 固定小数
                            std::cout << std::fixed << std::setprecision(4) << val; // 可调精度
                        }
                        //std::cout << std::scientific /*std::fixed*/ << std::setprecision(5) << (*data_)[pos];
                        if (i != N - 1) std::cout << ", ";
                    }
                }
                std::cout << "]";
                };

            std::cout << prefix;
            if (R == 0) {
                if (data() == nullptr) {
                    std::cout << "empty tensor!";
                }
                else {
                    double val = (*data_)[offset];
                    if ((val != 0.0) && (std::abs(val) < 1e-4 || std::abs(val) > 1e4)) {
                        // 科学计数法
                        std::cout << std::scientific << std::setprecision(5) << val;
                    }
                    else {
                        // 固定小数
                        std::cout << std::fixed << std::setprecision(4) << val; // 可调精度
                    }
                    //std::cout << (*data_)[offset]; // scalar
                }

            }
            else {
                print_dim(0, offset, idx, 0);
            }
            std::cout << std::endl;
        }

        Shape shape;
    private:
        Shape stride;
        Index offset = 0;
        std::shared_ptr<std::vector<T>> data_;

        // -------------------
        // 展开元素索引
        // -------------------

        template<typename... Args>
        Index ExpandIndices(Args... args) const {
            static_assert(sizeof...(Args) == sizeof...(args), "Wrong number of indices");
            Index pos = offset;
            Index idxs[] = { static_cast<Index>(args)... };
            for (size_t i = 0; i < sizeof...(args); ++i) {
                pos += idxs[i] * stride[i];
            }
            return pos;
        }


    };

    using Tensor = TensorT<float>;

}
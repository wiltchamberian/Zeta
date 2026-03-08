#pragma once
#include <vector>

namespace zeta {
    class Matrix {
    public:
        // 默认构造
        Matrix()
            : width(0), height(0) {
        }

        // 构造指定大小
        Matrix(int width, int height)
            : width(width), height(height)
        {
            data_.resize(width * height, 0.0);
        }

        // 拷贝构造
        Matrix(const Matrix& m)
            : width(m.width), height(m.height), data_(m.data_) {
        }

        // 拷贝赋值
        Matrix& operator= (const Matrix& m) {
            if (this != &m) {
                width = m.width;
                height = m.height;
                data_ = m.data_;
            }
            return *this;
        }

        // 移动构造
        Matrix(Matrix&& m) noexcept
            : width(m.width), height(m.height), data_(std::move(m.data_))
        {
            m.width = 0;
            m.height = 0;
        }

        // 移动赋值
        Matrix& operator= (Matrix&& m) noexcept {
            if (this != &m) {
                width = m.width;
                height = m.height;
                data_ = std::move(m.data_);

                m.width = 0;
                m.height = 0;
            }
            return *this;
        }

        // 访问
        void Set(int i, int j, float v) {
            data_[i * width + j] = v;
        }

        float* operator[] (int k) {
            return data_.data() + k * width;
        }

        void resize(int width, int height, float val = 0.0) {
            data_.resize(width * height, val);
            this->width = width;
            this->height = height;
        }

        int size() const {
            return height;
        }

        const float* data() const {
            return data_.data();
        }

        int width;
        int height;
        std::vector<float> data_;
    };

}
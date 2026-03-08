#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace zeta {
    class BinaryStream
    {
    public:
        BinaryStream(size_t initialCapacity = 1024)
        {
            buffer.resize(initialCapacity);
        }

        // =====================
        // 写入相关
        // =====================

        template<typename T>
        void write(const T& value)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                "Only POD supported");

            ensureCapacity(sizeof(T));
            std::memcpy(buffer.data() + writePos, &value, sizeof(T));
            writePos += sizeof(T);
        }

        void writeBytes(const void* src, size_t size);

        void writeString(const std::string& str);

        template<typename T>
        void writeVector(const std::vector<T>& vec)
        {
            size_t len = vec.size();
            write(len);
            writeBytes(vec.data(), sizeof(T) * len);
        }

        // =====================
        // 读取相关
        // =====================

        template<typename T>
        T read()
        {
            static_assert(std::is_trivially_copyable<T>::value,
                "Only POD supported");

            if (readPos + sizeof(T) > writePos)
                throw std::runtime_error("Read overflow");

            T value;
            std::memcpy(&value, buffer.data() + readPos, sizeof(T));
            readPos += sizeof(T);
            return value;
        }

        void readBytes(void* dst, size_t size);

        template<typename T>
        std::vector<T> readVector()
        {
            size_t len = read<size_t>();
            std::vector<T> vec(len);
            readBytes(vec.data(), sizeof(T) * len);
            return vec;
        }

        // =====================
        // 文件操作
        // =====================

        void saveToFile(const std::string& filename);

        void loadFromFile(const std::string& filename);

        void clear()
        {
            writePos = 0;
            readPos = 0;
        }

        size_t size() const { return writePos; }

    private:
        std::vector<char> buffer;
        size_t writePos = 0;
        size_t readPos = 0;

        void ensureCapacity(size_t extra);
    };
}
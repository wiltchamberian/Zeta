#pragma once

#include <vector>
#include <type_traits>

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

    void writeBytes(const void* src, size_t size)
    {
        ensureCapacity(size);
        std::memcpy(buffer.data() + writePos, src, size);
        writePos += size;
    }

    void writeString(const std::string& str)
    {
        size_t len = str.size();
        write(len);
        writeBytes(str.data(), len);
    }

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

    void readBytes(void* dst, size_t size)
    {
        if (readPos + size > writePos)
            throw std::runtime_error("Read overflow");

        std::memcpy(dst, buffer.data() + readPos, size);
        readPos += size;
    }

    //std::string readString()
    //{
    //    size_t len = read<size_t>();
    //    std::string str(len, '\0');
    //    readBytes(str.data(), len);
    //    return str;
    //}

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

    void saveToFile(const std::string& filename)
    {
        std::ofstream ofs(filename,
            std::ios::binary |
            std::ios::out |
            std::ios::trunc);

        if (!ofs)
            throw std::runtime_error("Failed to open file for writing");

        ofs.write(buffer.data(), writePos);
    }

    void loadFromFile(const std::string& filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        ifs.seekg(0);

        buffer.resize(size);
        ifs.read(buffer.data(), size);

        writePos = size;
        readPos = 0;
    }

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

    void ensureCapacity(size_t extra)
    {
        if (writePos + extra <= buffer.size())
            return;

        size_t newCap = buffer.size() == 0 ? 1024 : buffer.size();
        while (newCap < writePos + extra)
            newCap *= 2;

        buffer.resize(newCap);
    }
};
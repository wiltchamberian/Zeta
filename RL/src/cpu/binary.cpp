#include "binary.h"
#include <fstream>

namespace zeta {
    void BinaryStream::writeBytes(const void* src, size_t size)
    {
        ensureCapacity(size);
        std::memcpy(buffer.data() + writePos, src, size);
        writePos += size;
    }

    void BinaryStream::writeString(const std::string& str)
    {
        size_t len = str.size();
        write(len);
        writeBytes(str.data(), len);
    }

    void BinaryStream::readBytes(void* dst, size_t size)
    {
        if (readPos + size > writePos)
            throw std::runtime_error("Read overflow");

        std::memcpy(dst, buffer.data() + readPos, size);
        readPos += size;
    }

    void BinaryStream::saveToFile(const std::string& filename)
    {
        std::ofstream ofs(filename,
            std::ios::binary |
            std::ios::out |
            std::ios::trunc);

        if (!ofs)
            throw std::runtime_error("Failed to open file for writing");

        ofs.write(buffer.data(), writePos);
    }

    void BinaryStream::loadFromFile(const std::string& filename)
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

    void BinaryStream::ensureCapacity(size_t extra)
    {
        if (writePos + extra <= buffer.size())
            return;

        size_t newCap = buffer.size() == 0 ? 1024 : buffer.size();
        while (newCap < writePos + extra)
            newCap *= 2;

        buffer.resize(newCap);
    }
}

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>

class MnistLoader
{
public:
    struct Dataset
    {
        std::vector<std::vector<uint8_t>> images; // ц©уем╪ 28*28
        std::vector<uint8_t> labels;
        int rows = 0;
        int cols = 0;
    };

    static Dataset load(const std::string& imagePath,
        const std::string& labelPath);

private:
    static uint32_t readBigEndian(std::ifstream& file);
};

void mnist_test();
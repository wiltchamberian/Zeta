#include "mnist.h"
#include <random>
#include <filesystem>
#include <iostream>
#include "LeNet.h"

void saveAsPGM(const std::string& filename,
    const std::vector<uint8_t>& image,
    int rows,
    int cols)
{
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << cols << " " << rows << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), rows * cols);
    file.close();
}

uint32_t MnistLoader::readBigEndian(std::ifstream& file)
{
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return _byteswap_ulong(value);  // Windows
    // Linux: return __builtin_bswap32(value);
}

MnistLoader::Dataset MnistLoader::load(const std::string& imagePath,
    const std::string& labelPath)
{
    Dataset dataset;

    // ===========================
    // ∂¡»° labels
    // ===========================
    std::ifstream labelFile(labelPath, std::ios::binary);
    if (!labelFile)
        throw std::runtime_error("Cannot open label file");

    uint32_t magic = readBigEndian(labelFile);
    uint32_t size = readBigEndian(labelFile);

    if (magic != 2049)
        throw std::runtime_error("Label magic number mismatch");

    dataset.labels.resize(size);
    labelFile.read(reinterpret_cast<char*>(dataset.labels.data()), size);

    labelFile.close();

    // ===========================
    // ∂¡»° images
    // ===========================
    std::ifstream imageFile(imagePath, std::ios::binary);
    if (!imageFile)
        throw std::runtime_error("Cannot open image file");

    uint32_t imageMagic = readBigEndian(imageFile);
    uint32_t imageSize = readBigEndian(imageFile);
    uint32_t rows = readBigEndian(imageFile);
    uint32_t cols = readBigEndian(imageFile);

    if (imageMagic != 2051)
        throw std::runtime_error("Image magic number mismatch");

    dataset.rows = rows;
    dataset.cols = cols;

    dataset.images.resize(imageSize);

    for (uint32_t i = 0; i < imageSize; ++i)
    {
        dataset.images[i].resize(rows * cols);
        imageFile.read(reinterpret_cast<char*>(dataset.images[i].data()),
            rows * cols);
    }

    imageFile.close();

    return dataset;
}

void mnist_test() {
    std::cout << "Current path: "
        << std::filesystem::current_path()
        << std::endl;
    std::string root = std::filesystem::current_path().string();
    std::string trainImages = root + "\\..\\data\\mnist\\train-images.idx3-ubyte";
    std::string trainLabels = root + "\\..\\data\\mnist\\train-labels.idx1-ubyte";
    std::string testImages = root + "\\..\\data\\mnist\\t10k-images.idx3-ubyte";
    std::string testLabels = root + "\\..\\data\\mnist\\t10k-labels.idx1-ubyte";
    auto dataset = MnistLoader::load(trainImages, trainLabels);
    auto testset = MnistLoader::load(testImages, testLabels);

    std::cout << "Loaded: "
        << dataset.images.size()
        << " images\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dataset.images.size() - 1);

    for (int i = 0; i < 5; i++)
    {
        int r = dis(gen);

        std::cout << "Image [" << r << "] label = "
            << int(dataset.labels[r]) << std::endl;

        std::string filename =
            "mnist_sample_" + std::to_string(i) +
            "_label_" + std::to_string(dataset.labels[r]) +
            ".pgm";

        //saveAsPGM(filename,
        //    dataset.images[r],
        //    dataset.rows,
        //    dataset.cols);
    }
    
    //////train

    LeNet nn;
    nn.SetLearningRate(0.01);
    nn.createNetwork();

    int batchSize = 64;
    int epochs = 20;

    int numSamples = dataset.images.size();
    int numBatches = numSamples / batchSize;
    assert(dataset.cols == 28 && dataset.rows == 28);
    int siz = 28 * 28;

    std::vector<Tensor> batchData;
    std::vector<Tensor> labels;
    for (int b = 0; b < numBatches; b++) {
        int start = b * batchSize;
        int end = start + batchSize;
        Tensor images(batchSize, 1, 28, 28);
        float* d = images.data();
        for (int k = 0; k < batchSize; ++k) {
            std::transform(dataset.images[start + k].begin(),
                dataset.images[start + k].end(),
                d + k * siz,
                [](uint8_t val) { return val / 255.0f; });
        }
        batchData.push_back(images);
        Tensor label(batchSize, 10);
        float* l = label.data();
        for (int k = 0; k < batchSize; ++k) {
            int index = dataset.labels[start + k];
            label(k, index) = 1;
        }
        labels.push_back(label);
    }
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int b = 0; b < numBatches; b++) {
            nn.Forward(batchData[b]);
            nn.head->label = labels[b];
            nn.head->BindLabelToDevice();

            //nn.fc2->FetchResultToCpu();
            nn.Backward();

            //nn.fc2->FetchResultToCpu();

            //nn.ErrorCheck();
            if (b % 100 == 0) {
                auto loss = nn.head->FetchLoss();
                std::cout << "eoch:" << epoch << "loss:" << loss << std::endl;
            }

            nn.Step();

            
        }
    }

    //test
    int correct = 0;
    for (int i = 0; i < testset.labels.size()/batchSize; ++i) {
        Tensor input(batchSize, 1, 28, 28);
        float* d = input.data();
        for (int k = 0; k < batchSize; ++k) {
            int index = i * batchSize + k;
            std::transform(testset.images[index].begin(),
                testset.images[index].end(),
                d + k * siz,
                [](uint8_t val) { return val / 255.0f; });
        }
        nn.Forward(input);
        nn.head->FetchActivationToCpu();
        
        for (int j = 0; j < batchSize; ++j) {
            int index = i * batchSize + j;
            int maxIndex = -1;
            float max = -INFINITY;
            for (int t = 0; t < 10; ++t) {
                float p = nn.head->distribution(j, t);
                if (p > max) {
                    max = p;
                    maxIndex = t;
                }
            }
            if (testset.labels[index] == maxIndex) {
                correct += 1;
            }
        }
    }

    std::cout << "accuracy:" << ((float)correct / float(testset.labels.size())) << std::endl;
}
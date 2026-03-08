#include "mnist.h"
#include <random>
#include <filesystem>
#include <iostream>
#include "LeNet.h"

using namespace zeta;

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

    //////train
    LeNet nn;
    nn.SetLearningRate(0.01);
    //nn.c = 0.0001;
    nn.createDnnNetwork();

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

    auto epoch_start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int b = 0; b < numBatches; b++) {
            nn.Forward(batchData[b]);
            nn.head->label = labels[b];
            nn.head->BindLabelToDevice();

            //nn.c2->PrintActivation();
            //nn.fc->PrintActivation();
            //nn.c1->PrintActivation();
            //nn.fc2->PrintActivation();

            nn.Backward();
            
            //if (epoch == 0 && b == 0) {
            //    nn.FetchGrad();
            //    //nn.c1->PrintGrad();
            //    //nn.c2->PrintGrad();
            //    //nn.PrintGrad();
            //    nn.fc2->PrintGrad();
            //    Tensor loss = nn.head->FetchLoss();
            //    loss.print_torch_style();
            //    break;
            //}
            //nn.fc2->FetchResultToCpu();

            //nn.ErrorCheck();

            /*nn.fc2->PrintActivation();
            nn.head->label.print_torch_style("label:");
            nn.head->PrintActivation();
            nn.fc2->PrintDelta();
            nn.fc2->PrintBGrad();

            nn.fc->PrintDelta();
            nn.fc->PrintBGrad();*/

            //nn.c2->PrintWGrad();
            //nn.c2->PrintBGrad();

            /*nn.c1->PrintWGrad();
            nn.c1->PrintBGrad();

            auto loss = nn.head->FetchLoss();
            loss.print_torch_style();
            std::cout << std::endl;*/
            if (b % 100 == 0) {
                auto loss = nn.head->FetchLoss();
                std::cout << "Epoch " << epoch << "Batch " << b;
                loss.print_torch_style(", Loss ");
            }

            nn.Step();

            
        }
    }
    auto epoch_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
    std::cout << " finished, total time: "
        << std::fixed << std::setprecision(3)
        << epoch_duration.count() << " s" << std::endl;

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
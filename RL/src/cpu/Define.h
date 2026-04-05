#pragma once

//used for padding and stride
struct Size2D {
    int h;
    int w;
};

//Must not change order, new items add to tail
//otherwise can't not read old files
enum class LayerType {
    Basic,
    Fully,
    Conv,
    Activation,
    Act_Relu,
    Act_Tanh,
    Act_Sigmoid,
    Act_ClippedRelu,
    Act_Elu,
    Act_Identity,
    Act_SWISH,
    Mse,
    Softmax,
    Add,
    Output,
    MaxPooling
};


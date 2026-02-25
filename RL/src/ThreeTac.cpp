#include "ThreeTac.h"
#include "reluLayer.h"
#include "tanhLayer.h"

void ThreeTacState::Init() {
    for (int i = 0; i < 9; ++i) {
        board[i] = 0;
    }
}

Tensor ThreeTacState::Encode() const {
    Tensor result(1, 2, 3, 3);
    float* d = result.data();
    for (int i = 0; i < 9; ++i) {
        if (board[i] == player) {
            d[i] = 1;
        }
        else {
            d[i] = 0;
        }
    }
    for (int i = 0; i < 9; ++i) {
        if (board[i] == -player) {
            d[9 + i] = 1;
        }
        else {
            d[9 + i] = 0;
        }
    }
    return result;
}

void ThreeTacState::FromTensor(const Tensor& result) {
    player = 1;
    const float* d = result.data();
    for (int i = 0; i < 9; ++i) {
        if (d[i] == 1) {
            board[i] = 1;
        }
        else {
            board[i] = 0;
        }
    }
    for (int i = 0; i < 9; ++i) {
        if (d[9 + i] == 1) {
            board[i] = -1;
        }
    }
}

std::vector<int> ThreeTacState::legalActions() const {
    std::vector<int> actions;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == 0) {
            actions.push_back(i);
        }
    }
    return actions;
}

std::unique_ptr < mcts::State > ThreeTacState::next_state(int action) const {
    std::unique_ptr <ThreeTacState> st = std::make_unique<ThreeTacState>();
    *st = *this;
    st->board[action] = player;
    st->player = -st->player;
    st->depth += 1;
    return st;
}

bool ThreeTacState::is_terminal() const
{
    int opp = -player;   // 对方棋子

    // 8 条可能的三连
    static const int win_lines[8][3] = {
        {0,1,2}, {3,4,5}, {6,7,8},      // 行
        {0,3,6}, {1,4,7}, {2,5,8},      // 列
        {0,4,8}, {2,4,6}                // 对角线
    };

    for (int i = 0; i < 8; ++i)
    {
        if (board[win_lines[i][0]] == opp &&
            board[win_lines[i][1]] == opp &&
            board[win_lines[i][2]] == opp)
        {
            return true;
        }
    }

    bool full = true;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == 0) {
            full = false;
        }
    }
    if (full) {
        return true;
    }

    return false;
}

float ThreeTacState::terminal_value() const {
    int opp = -player;   // 对方棋子

    // 8 条可能的三连
    static const int win_lines[8][3] = {
        {0,1,2}, {3,4,5}, {6,7,8},      // 行
        {0,3,6}, {1,4,7}, {2,5,8},      // 列
        {0,4,8}, {2,4,6}                // 对角线
    };

    for (int i = 0; i < 8; ++i)
    {
        if (board[win_lines[i][0]] == opp &&
            board[win_lines[i][1]] == opp &&
            board[win_lines[i][2]] == opp)
        {
            return -1;
        }
    }

    return 0;
}

char ThreeTacState::character(int p) const {
    static char chs[3] = { 'O','.' ,'X' };
    return chs[p + 1];
}

void ThreeTacState::printState() const {

    std::cout << character(board[6]) << " | ";
    std::cout << character(board[7]) << " | ";
    std::cout << character(board[8]) << std::endl;

    std::cout << character(board[3]) << " | ";
    std::cout << character(board[4]) << " | ";
    std::cout << character(board[5]) << std::endl;

    std::cout << character(board[0]) << " | ";
    std::cout << character(board[1]) << " | ";
    std::cout << character(board[2]) << std::endl;

    if (player == 1) {
        std::cout << "player:human" << std::endl;
    }
    else {
        std::cout << "player:AI" << std::endl;
    }

}


std::shared_ptr<mcts::State> ThreeTacProxy::createState() {
    auto state = std::make_shared<ThreeTacState>();
    state->Init();
    return state;
}

void ThreeTacProxy::createNetwork(float learningRate) {
    nn = std::make_unique<CuNN>();
    nn->SetLearningRate(learningRate);

    auto fc = nn->CreateLayer<Linear>(18, 9);
    auto relu = nn->CreateLayer<CuReluLayer>();

    auto fc2 = nn->CreateLayer<Linear>(9, 9);
    auto sf = nn->CreateLayer<CuSoftmaxCrossEntropyLayer>();

    auto fc3 = nn->CreateLayer<Linear>(9, 3);
    auto relu2 = nn->CreateLayer<CuReluLayer>();
    auto fc4 = nn->CreateLayer<Linear>(3, 1);
    auto tanh = nn->CreateLayer<CuTanhLayer>();
    auto mse = nn->CreateLayer<CuMseLayer>();

    auto add = nn->CreateLayer<CuAddLayer>();

    fc->AddLayer(relu)->AddLayer(fc2)->AddLayer(sf)->AddLayer(add);
    relu->AddLayer(fc3)->AddLayer(relu2)->AddLayer(fc4)->AddLayer(tanh)->AddLayer(mse)->AddLayer(add);

    root = fc;
    policyHead = sf;
    valueHead = mse;

    nn->AllocDeviceMemory();
}
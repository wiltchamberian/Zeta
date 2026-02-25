#include "TicTac.h"
#include "reluLayer.h"
#include "tanhLayer.h"
#include <algorithm>
#include <random>

std::unique_ptr<mcts::State> TicTac::next_state(int action) const
{
    std::unique_ptr<TicTac> s = std::make_unique<TicTac>();
    *s = *this;

    //action
    int playerId = action / 3;
    int targetId = action % 3;
    int playerRover = 0;
    int targetRover = 0;
    int count = 0;

    int startPos = 0;
    int endPos = 0;
    for (int i = 0; i < 9; ++i)
    {
        if (s->board[i] == player) {
            if (playerRover == playerId) {
                startPos = i;
                count += 1;
            }
            playerRover++;
        }
        if (s->board[i] == 0) {
            if (targetRover == targetId) {
                endPos = i;
                count += 1;
            }
            targetRover++;
        }
        if (count == 2) {
            break;
        }
    }

    s->board[startPos] = 0;
    s->board[endPos] = player;
    
    s->player = -player;
    s->depth++;
    return s;
}

bool TicTac::legal(int i, int j) const {
    if (i == 4 || j == 4) {
        return true;
    }
    else if (i == 0) {
        if (j == 1 || j == 3) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 1) {
        if (j == 0 || j == 2 ) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 2) {
        if (j == 1 || j == 5) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 3) {
        if (j == 0  || j == 6) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 5) {
        if (j == 2  || j == 8) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 6) {
        if (j == 3 || j == 7) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 7) {
        if ( j == 6 || j == 8) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 8) {
        if (j == 5 || j == 7) {
            return true;
        }
        else {
            return false;
        }
    }
    return false;
}

bool TicTac::legalAction(int action) const {
    int playerId = action / 3;
    int targetId = action % 3;

    int playerRover = 0;
    int targetRover = 0;
    int count = 0;
    int playerPos = 0;
    int targetPos = 0;
    for (int i = 0; i < 9; ++i)
    {
        if (board[i] == player) {
            if (playerRover == playerId) {
                playerPos = i;
                count += 1;
            }
            playerRover++;
        }
        if (board[i] == 0) {
            if (targetRover == targetId) {
                targetPos = i;
                count += 1;
            }
            targetRover++;
        }
        if (count == 2) {
            break;
        }
    }

    return legal(playerPos, targetPos);
}

std::vector<int> TicTac::legalActions() const{
    std::vector<int> actions;
    int m = 0;
    int n = 0;
    int startPos = 0;
    int endPos = 0;
    std::vector<int> startPositions;
    std::vector<int> endPositions;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == player) {
            startPositions.push_back(i);
        }
        else if (board[i] == 0) {
            endPositions.push_back(i);
        }
    }
    for (int i = 0; i < 9; ++i) {
        int s = i / 3;
        int t = i % 3;
        if (legal(startPositions[s], endPositions[t])) {
            actions.push_back(i);
        }
    }
    
    return actions;
}


std::shared_ptr<mcts::State> TicTacProxy::createState() {
    std::shared_ptr<mcts::State> st = std::make_shared<TicTac>();
    st->Init();
    return st;
}

CuHead TicTacProxy::predict(const mcts::State* state) {
    CuHead head;
    Tensor input = state->Encode();

    nn->Forward(input);
    policyHead->FetchActivationToCpu();
    Tensor& result = policyHead->distribution;
    valueHead->FetchPredYToCpu();
    Tensor& value = valueHead->predY;

    std::vector<int> legalActions = state->legalActions();

    head.value = value(0, 0);
    for (int i = 0; i < legalActions.size(); ++i) {
        head.policy.push_back(result(0, legalActions[i]));
    }

    //re-softmax
    float total = 0.0f;
    for (int j = 0; j < head.policy.size(); ++j) {
        total += head.policy[j];
    }
    if (total > 0) {
        for (int j = 0; j < head.policy.size(); ++j) {
            head.policy[j] = head.policy[j] / total;
        }
    }
    return head;

}

void TicTacProxy::setLearningRate(float rate) {
    nn->SetLearningRate(rate);
}

//build from scratch
void TicTacProxy::createNetwork(float learningRate) {

    nn = std::make_unique<CuNN>();
    nn->SetLearningRate(learningRate);

    //layer
    auto c1 = nn->CreateLayer<CuConvolutionLayer>(16, 2, 3, 3);
    c1->alpha = 0.0;
    c1->padH = 1;
    c1->padW = 1;
    root = c1;

    auto c2 = nn->CreateLayer<CuConvolutionLayer>(16, 16, 3, 3);
    c2->alpha = 0.0;
    c2->padH = 1;
    c2->padW = 1;
    c1->AddLayer(c2);

    //1d conv
    auto c3 = nn->CreateLayer<CuConvolutionLayer>(1, 16, 1, 1);
    c3->padH = 0;
    c3->padW = 0;
    c2->AddLayer(c3);

    auto relu = nn->CreateLayer<CuReluLayer>();
    c3->AddLayer(relu);

    auto fully1 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    auto relu1 = nn->CreateLayer<CuReluLayer>();
    fully1->AddLayer(relu1);

    auto cross = nn->CreateLayer<CuSoftmaxCrossEntropyLayer>();
    relu1->AddLayer(cross);

    auto fully2 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    auto relu2 = nn->CreateLayer<CuReluLayer>();
    fully2->AddLayer(relu2);


    auto fully2_1 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 1);

    relu2->AddLayer(fully2_1);

    auto tanh = nn->CreateLayer<CuTanhLayer>();
    fully2_1->AddLayer(tanh);

    auto mse = nn->CreateLayer<CuMseLayer>();

    tanh->AddLayer(mse);

    relu->AddLayer(fully1);
    relu->AddLayer(fully2);

    auto tail = nn->CreateLayer<CuAddLayer>();
    cross->AddLayer(tail);
    mse->AddLayer(tail);

    policyHead = cross;
    valueHead = mse;

    nn->AllocDeviceMemory();
}

void TicTacProxy::train(const std::vector<mcts::Entry>& entries) {
    if (entries.empty()) {
        return;
    }
    Tensor label(entries.size(), entries[0].label.shape[0]);
    Tensor values(entries.size());
    Tensor states(entries.size(), entries[0].state.shape[1], entries[0].state.shape[2], entries[0].state.shape[3]);
    for (int i = 0; i < entries.size(); ++i) {
        label[i].copy(entries[i].label);
        values(i) = entries[i].value;
        states[i].copy(entries[i].state);
    }
    policyHead->label = label;
    valueHead->label = values;

    nn->Forward(states);

    policyHead->BindLabelToDevice();
    valueHead->BindLabelToDevice();


    float crossLoss = policyHead->FetchLoss();
    float mseLoss = valueHead->FetchLoss();
    float loss = crossLoss + mseLoss;
    std::cout << "loss:" << loss << " mse:" << mseLoss << " cross:" << crossLoss << std::endl;



    nn->Backward();
    nn->Step();
}






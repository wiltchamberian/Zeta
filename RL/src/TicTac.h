#pragma once
#include <memory>
#include <vector>
#include <random>
#include "MctsAlgo.h"

struct Action {
    Action() {}
    Action(char a, char b):startPos(a),endPos(b) {
    }
    char startPos = 0;
    char endPos = 0;
};

class TicTac: public mcts::State {
public:
    TicTac() {
        memset(board, 0, 9 * sizeof(char));
        board[0] = 1;
        board[1] = 1;
        board[2] = 1;
        board[6] = -1;
        board[7] = -1;
        board[8] = -1;
    }
    char board[9];
    void Init() override {
        memset(board, 0, 9 *sizeof(char));
        board[0] = 1;
        board[1] = 1;
        board[2] = 1;
        board[6] = -1;
        board[7] = -1;
        board[8] = -1;
    }

    Tensor Encode() const {
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
    void FromTensor(const Tensor& result) {
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

    bool legal(int i, int j) const;
    bool legalAction(int action) const;
    std::vector<int> legalActions() const override;
    std::shared_ptr<mcts::State> next_state(int action) const;
    bool is_terminal() const
    {
        if (board[4] == (-player)) {
            for (int i = 0; i < 4; ++i) {
                if (board[i] == (-player) && board[8 - i] == (-player)) {
                    return true;
                }
            }
        }
        return false;
    }

    char character(int p) const {
        static char chs[3] = { 'O','.' ,'X' };
        return chs[p + 1];
    }

    void printState() const {

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

    float terminal_value() const
    {
        return -1.0f;
    }

    int winner() const override {
        return -player;
    }
};

class TicTacProxy : public mcts::Proxy {
public:
    std::unique_ptr<CuNN> nn = nullptr;

    virtual std::shared_ptr<mcts::State> createState();
    std::vector<mcts::Entry> createSamples();
    CuHead predict(const mcts::State* state);
    void setLearningRate(float rate);
    void createNetwork(float learningRate);
    void train(const std::vector<mcts::Entry>& entries);
    virtual Proxy* Clone() const override;
    CuLayer* root = nullptr;
    CuSoftmaxCrossEntropyLayer* policyHead = nullptr;
    CuMseLayer* valueHead = nullptr;
};


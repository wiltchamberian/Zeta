#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <stack>
#include "TicTac.h"


#define HEX_WIDTH 4
#define HEX_DIM 16

class HexState : public zeta::State {
public:
    char board[HEX_DIM]; // 0=空, 1=player1, -1=player2

    HexState();

    void Init() override;

    int idx(int x, int y) const;

    bool inBoard(int x, int y) const;

    virtual Tensor Encode() const;
    virtual void FromTensor(const Tensor& result);

    std::vector<int> legalActions() const override;

    std::shared_ptr<State> next_state(int action) const override;

    bool is_terminal() const override;

    float terminal_value() const override;

    int winner() const override;

    uint64_t Hash() const override;
    void UnHash(uint64_t hash) override;

    // =========================
    // ⭐ 核心：DFS判断连通
    // =========================
    bool checkWin(int p) const;

    void printState() const override;

    uint64_t currentHash = 0;
};

class HexProxy : public TicTacProxy {
public:
    using StateType = HexState;
    HexProxy() {}
    virtual CuHead predict(const State* state)  override;
    virtual std::shared_ptr<State> createState() const override;
    virtual void createNetwork(float learningRate) override;
    virtual void train(const Tensor& states, const Tensor& actions);
    
};


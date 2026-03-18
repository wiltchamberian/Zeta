#pragma once
#include "TicTac.h"

using namespace zeta;

#define BOARD_WIDTH 5

class Go :public mcts::State {
public:
    static const int BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;
    virtual void Init() override;
    virtual Tensor Encode() const override;
    virtual void FromTensor(const Tensor& result) override;
    virtual std::vector<int> legalActions() const override;
    virtual std::shared_ptr<mcts::State> next_state(int action) const override;
    virtual bool is_terminal() const override;
    virtual float terminal_value() const override;
    virtual void printState() const override;
    virtual int winner() const override;
    char board[BOARD_SIZE];

    char character(int p) const;
    int countLiberties(int i, int j) const;
    void removeGroup(int i, int j);

    int continuous_pass = 0;
};

class GoProxy : public TicTacProxy {
public:
    using StateType = Go;
    virtual std::shared_ptr<mcts::State> createState();
    void createNetwork(float learningRate);
    virtual Proxy* Clone() const override;
};
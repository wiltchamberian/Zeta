#pragma once
#include "TicTac.h"

class ThreeTacState : public mcts::State{
public:
    virtual void Init() override;
    virtual Tensor Encode() const override;
    virtual void FromTensor(const Tensor& result) override;
    virtual std::vector<int> legalActions() const override;
    virtual std::shared_ptr<mcts::State> next_state(int action) const override;
    virtual bool is_terminal() const override;
    virtual float terminal_value() const override;
    virtual void printState() const override;
    virtual int winner() const override;
    char character(int p) const;

protected:
    char board[9];
    int win = 0;

};

class ThreeTacProxy : public TicTacProxy {
public:
    using StateType = ThreeTacState;
    virtual std::shared_ptr<mcts::State> createState() const override;
    void createNetwork(float learningRate);
    virtual Proxy* Clone() const override;
};




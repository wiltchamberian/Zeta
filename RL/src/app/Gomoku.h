#pragma once
#include "TicTac.h"

#define GOMOKU_W 9
#define GOMOKU_DIM 81
#define GOMOKU_WIN 5

class Gomoku : public State {
public:
    virtual void Init() override;
    virtual Tensor Encode() const override;
    virtual void FromTensor(const Tensor& result) override;
    virtual std::vector<int> legalActions() const override;
    virtual std::shared_ptr<State> next_state(int action) const override;
    virtual bool is_terminal() const override;
    virtual float terminal_value() const override;
    virtual void printState() const override;
    virtual int winner() const override;
    uint64_t Hash() const override;
    void UnHash(uint64_t hash) override;
    char character(int p) const;

    
protected:
    uint64_t currentHash = 0;

    bool check_win(int x, int y) const;
    char board[GOMOKU_DIM];
    int win = 0;
    int lastAction = 0;
};

class GomokuProxy : public TicTacProxy {
public:
    using StateType = Gomoku;
    virtual std::shared_ptr<State> createState() const override;
    void createNetwork(float learningRate);
    void createNNnetwork(float l, OptimizerType optType);
    virtual Proxy* Clone() const override;

};

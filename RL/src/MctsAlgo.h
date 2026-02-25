#pragma once


#include <memory>
#include <vector>
#include <random>
#include "CuNN.h"
#include "tensor.h"

namespace mcts {

    struct Entry {
        Tensor label;
        Tensor state;
        float value;
    };

    class State {
    public:
        // 默认构造
        State() = default;

        // 拷贝构造
        State(const State& other)
            : player(other.player),
            depth(other.depth)
        {
        }

        // 移动构造
        State(State&& other) noexcept
            : player(other.player),
            depth(other.depth)
        {
        }

        // 拷贝赋值
        State& operator=(const State& other)
        {
            if (this != &other)
            {
                player = other.player;
                depth = other.depth;
            }
            return *this;
        }

        // 移动赋值
        State& operator=(State&& other) noexcept
        {
            if (this != &other)
            {
                player = other.player;
                depth = other.depth;
            }
            return *this;
        }

        // 必须是 virtual 析构函数（关键）
        virtual ~State() = default;
        
        virtual void Init() {}
        virtual Tensor Encode() const { return Tensor(1);  }
        virtual void FromTensor(const Tensor& result) {}
        virtual std::vector<int> legalActions() const { return {};  }
        virtual std::unique_ptr<State> next_state(int action) const { return nullptr; }
        virtual bool is_terminal() const { return false; }
        virtual float terminal_value() const { return 0; }
        virtual void printState() const {}
        int player = 1;
        int depth = 0;
    };

    class Proxy {
    public:
        virtual std::shared_ptr<mcts::State> createState() { return nullptr; }
        virtual CuHead predict(const mcts::State* state) { return CuHead(); }
        virtual void setLearningRate(float rate) {}
        virtual void createNetwork(float learningRate) {}
        virtual void train(const std::vector<mcts::Entry>& entries) {}

    };

    //each edge save a prior probaliby P(s,a)
    //visit count N(s,a) and action-value Q(s,a)
    struct Edge {
        Edge(int ac, float policy)
            :action(ac)
            , prior(policy) {

        }
        int action = 0;

        float prior = 0.0f;
        int visit_count = 0;
        float W = 0.0f;

        float Q() {
            if (visit_count == 0) {
                return 0;
            }
            else {
                return W / visit_count;
            }
        }
    };

    class Node {
    public:
        Edge* parentEdge = nullptr;
        Node* parent = nullptr;
        std::shared_ptr<State> state;
        bool expanded = false;
        std::vector<std::unique_ptr<Edge>> edges;
        std::vector<std::unique_ptr<Node>> children;

        std::vector<float> getPolicyDistribution(float);
        int subTreeDepth = 0;
    };

    
    class ReplayBuffer {
    public:
        std::vector<Entry> entries;

        std::vector<Entry> sample(size_t batch_size);
    };

    struct Setting {
        int simulationCount = 0;
        int num_episodes = 100;
        int trainStepsPerEpisode = 20;
        int batchSize = 100;
        int miniBatchSize = 32;
        float c_puct = 1.0;
        int maxChessLength = 20;
    };

    class Mcts {
    public:
        void backTrace(Node* n, float value);
        void simulate(Node* n);
        //一次搜索包含多次mcts simulation
        void search();
        //一次完整对弈，每一步棋包含一次搜索
        void selfPlay(ReplayBuffer& buffer);
        //一次训练
        void train();
        //实战下棋
        std::unique_ptr<State> play(const std::unique_ptr<State>& state) const;
        //随机下棋
        std::unique_ptr<State> randomPlay(const std::unique_ptr<State>& state) const;
        void InitRandom();
        void InitRandom(uint32_t seed);

        Proxy* proxy = nullptr;

        std::mt19937 gen;

        Setting setting;
    };

}




#pragma once
#include "tensor.h"
#include "binary.h"
#include "CuNN.h"
#include "Help.h"


namespace zeta {

    

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
        virtual Tensor Encode() const { return Tensor(1); }
        virtual void FromTensor(const Tensor& result) {}
        virtual std::vector<int> legalActions() const { return {}; }
        virtual std::shared_ptr<State> next_state(int action) const { return nullptr; }
        virtual bool is_terminal() const { return false; }
        virtual float terminal_value() const { return 0; }
        virtual void printState() const {}
        virtual int winner() const {
            return 0;
        }
        virtual uint64_t Hash() const { return 0; }
        virtual void UnHash(uint64_t hash) { return; }
        virtual std::vector<std::shared_ptr<State>> permuteStates(const std::vector<double>& policy, std::vector<std::vector<double>>& policies) { return {}; }

        int player = 1;
        int depth = 0;
    };

    class Proxy {
    public:
        virtual ~Proxy() {}
        virtual std::shared_ptr<State> createState() const { return nullptr; }
        virtual void Save(const std::string& path) const {}
        virtual CuHead predict(const State* state) { return CuHead(); }
        virtual CuHead predict_s(const State* state) { return CuHead(); }
        virtual void setLearningRate(float rate) {}
        virtual void createNetwork(float learningRate) {}
        virtual void train(const std::vector<Entry>& entries) {}
        virtual void train(const Tensor& states, const Tensor& actions, const Tensor& values) {}
        virtual Proxy* Clone() const = 0;
        virtual void Save(BinaryStream& stream) const {}
        virtual void Load(BinaryStream& stream) {}
        virtual void PrintLoss() const {}
        int totalActionCount = 0;
        int version = 0;
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
        float q = 0.0f;
        float W = 0.0f;
        float reward = 0;


        float Q() {
            if (visit_count == 0) {
                return 0;
            }
            else {
                return W / visit_count;
            }
        }
    };




    //alloc on memory pool
    class Node {
    public:
        friend class NodePool<Node>;
        ~Node() {
            for (int i = 0; i < children.size(); ++i) {
                if (children[i] != nullptr) {
                    delete children[i];
                }
            }
        }
    private:
        Node() {

        }

    public:
        Edge* parentEdge = nullptr;
        Node* parent = nullptr;
        std::shared_ptr<State> state;
        bool expanded = false;
        std::vector<std::unique_ptr<Edge>> edges;
        std::vector<Node*> children;

        std::vector<double> getPolicyDistribution(double, int totalActionCount);
        int subTreeDepth = 0;
    };

    class Agent {
    public:
        
    };
}
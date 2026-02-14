#include <vector>
#include <atomic>
#include <memory>
#include <cmath>
#include <thread>
#include <mutex>
#include <random>
#include <iostream>

//////////////////////////////////////////////////////////////
// 示例 State
//////////////////////////////////////////////////////////////

struct State
{
    static constexpr int N = 5;
    static constexpr int MaxDepth = 50;
    char board[N*N] = {};

    char player = 1;//black, -1:white
    int depth = 0;

    std::vector<char> legal_actions() const
    {
        std::vector<char> result;
        if (depth >= MaxDepth) return {};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (board[i * N +j ] == 0) {
                    result.push_back(i * N + j);
                }
            }
        }
        return result;
    }

    State next_state(int action)
    {
        State s = *this;
        board[action] = player;
        s.player = -player;
        s.depth++;
        return s;
    }

    bool is_terminal() const
    {
        return depth >= 6;
    }

    float terminal_value() const
    {
        return 0.0f;
    }
};

//////////////////////////////////////////////////////////////
// 网络接口
//////////////////////////////////////////////////////////////

struct NetworkOutput
{
    std::vector<float> policy;
    float value;
};

class NeuralNetworkProxy
{
public:
    NetworkOutput predict(const State& state)
    {
        NetworkOutput out;
        auto acts = state.legal_actions();
        int n = acts.size();
        out.policy.resize(n, 1.0f / std::max(1, n));
        out.value = 0.0f;
        return out;
    }
};

//////////////////////////////////////////////////////////////
// Edge
//////////////////////////////////////////////////////////////

struct Edge
{
    int action;
    float prior;

    std::atomic<int> visit_count{ 0 };
    std::atomic<float> value_sum{ 0.0f };

    Edge(int a, float p) : action(a), prior(p) {}

    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;

    float Q() const
    {
        int n = visit_count.load(std::memory_order_relaxed);
        if (n == 0) return 0.0f;
        return value_sum.load(std::memory_order_relaxed) / n;
    }
};

//////////////////////////////////////////////////////////////
// Node（vector 加锁版本）
//////////////////////////////////////////////////////////////

struct Node
{
    State state;
    bool expanded = false;
    float value = 0.0f;

    std::vector<std::unique_ptr<Edge>> edges;
    std::vector<std::unique_ptr<Node>> children;

    std::mutex mtx; // vector 扩展锁

    Node(const State& s) : state(s) {}
};

//////////////////////////////////////////////////////////////
// MCTS
//////////////////////////////////////////////////////////////

class MCTS
{
public:
    MCTS(int simulations = 800, int threads = 4, float cpuct = 1.5f, float virtual_loss = 3.0f)
        : simulations_(simulations),
        threads_(threads),
        cpuct_(cpuct),
        virtual_loss_(virtual_loss)
    {
    }

    int search(const State& root_state, NeuralNetworkProxy& network)
    {
        root_ = std::make_unique<Node>(root_state);

        std::vector<std::thread> workers;

        for (int t = 0; t < threads_; ++t)
        {
            workers.emplace_back([this]()
                {
                    int sims = simulations_ / threads_;
                    for (int i = 0; i < sims; ++i)
                    {
                        simulate(root_.get());
                    }
                });
        }

        for (auto& th : workers)
            th.join();

        return select_action();
    }

    std::vector<float> get_policy_distribution(Node* node, float temperature = 1.0f);

    std::unique_ptr<Node> root_;
private:

    
    NeuralNetworkProxy network_;
    int simulations_;
    int threads_;
    float cpuct_;
    float virtual_loss_;

    //////////////////////////////////////////////////////////////

    float simulate(Node* node)
    {
        if (node->state.is_terminal())
            return node->state.terminal_value();

        {
            std::lock_guard<std::mutex> lock(node->mtx);
            if (!node->expanded)
            {
                return expand(node);
            }
        }

        int best = select_child(node);
        Edge* edge = nullptr;

        {
            std::lock_guard<std::mutex> lock(node->mtx);
            edge = node->edges[best].get();
        }

        // 虚拟损失
        edge->visit_count.fetch_add(1);
        edge->value_sum.fetch_sub(virtual_loss_);

        Node* child = nullptr;

        {
            std::lock_guard<std::mutex> lock(node->mtx);
            if (!node->children[best])
            {
                State next = node->state.next_state(edge->action);
                node->children[best] = std::make_unique<Node>(next);
            }
            child = node->children[best].get();
        }

        float value = simulate(child);

        edge->value_sum.fetch_add(virtual_loss_);
        edge->value_sum.fetch_add(value);

        return -value;
    }

    //////////////////////////////////////////////////////////////

    float expand(Node* node)
    {
        auto output = network_.predict(node->state);
        auto actions = node->state.legal_actions();

        {
            std::lock_guard<std::mutex> lock(node->mtx);
            node->expanded = true;

            node->edges.reserve(actions.size());
            node->children.resize(actions.size());

            for (size_t i = 0; i < actions.size(); ++i)
            {
                node->edges.push_back(std::make_unique<Edge>(actions[i], output.policy[i]));
                node->children[i] = nullptr;
            }
        }

        return output.value;
    }

    //////////////////////////////////////////////////////////////

    int select_child(Node* node)
    {
        float total = 0;
        {
            std::lock_guard<std::mutex> lock(node->mtx);
            for (auto& e : node->edges)
                total += e->visit_count.load();
        }

        float best_score = -1e9f;
        int best = 0;

        {
            std::lock_guard<std::mutex> lock(node->mtx);
            for (size_t i = 0; i < node->edges.size(); ++i)
            {
                Edge* e = node->edges[i].get();
                float q = e->Q();
                float u = cpuct_ * e->prior * std::sqrt(total + 1) / (1 + e->visit_count.load());
                float score = q + u;
                if (score > best_score)
                {
                    best_score = score;
                    best = i;
                }
            }
        }

        return best;
    }

    //////////////////////////////////////////////////////////////

    int select_action()
    {
        int best_action = -1;
        int best_visits = -1;

        {
            std::lock_guard<std::mutex> lock(root_->mtx);
            for (auto& e : root_->edges)
            {
                int visits = e->visit_count.load();
                if (visits > best_visits)
                {
                    best_visits = visits;
                    best_action = e->action;
                }
            }
        }

        return best_action;
    }
};




struct ReplayEntry
{
    State state;
    std::vector<float> policy; // MCTS visit count -> 归一化概率
    float value;
};

class ReplayBuffer
{
public:
    ReplayBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity), index_(0), size_(0) {
    }

    void push(const State& state, const std::vector<float>& policy, float value)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_[index_] = { state, policy, value };
        index_ = (index_ + 1) % capacity_;
        if (size_ < capacity_) size_++;
    }

    // 随机采样
    std::vector<ReplayEntry> sample(size_t batch_size)
    {
        std::vector<ReplayEntry> batch;
        batch.reserve(batch_size);
        std::lock_guard<std::mutex> lock(mtx_);

        std::uniform_int_distribution<size_t> dist(0, size_ - 1);
        std::mt19937 rng(std::random_device{}());

        for (size_t i = 0; i < batch_size; ++i)
        {
            batch.push_back(buffer_[dist(rng)]);
        }
        return batch;
    }

    size_t size() const { return size_; }

private:
    size_t capacity_;
    std::vector<ReplayEntry> buffer_;
    size_t index_;
    size_t size_;
    mutable std::mutex mtx_;
};


void self_play_episode(std::shared_ptr<NeuralNetworkProxy> network_main, ReplayBuffer& replay);

extern int runMcts();

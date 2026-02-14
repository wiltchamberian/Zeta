#include "mcts.h"

std::vector<float> MCTS::get_policy_distribution(Node* node, float temperature)
{
    std::vector<float> pi;
    {
        std::lock_guard<std::mutex> lock(node->mtx);
        size_t n_actions = node->edges.size();
        pi.resize(n_actions, 0.0f);

        float sum = 0.0f;
        for (size_t i = 0; i < n_actions; ++i)
        {
            float v = std::pow(float(node->edges[i]->visit_count.load()), 1.0f / temperature);
            pi[i] = v;
            sum += v;
        }

        if (sum > 0)
        {
            for (size_t i = 0; i < n_actions; ++i)
                pi[i] /= sum;
        }
    }
    return pi;
}

//////////////////////////////////////
std::mutex network_mtx; // 用于拷贝主网络

void self_play_episode(std::shared_ptr<NeuralNetworkProxy> network_main, ReplayBuffer& replay)
{
    // 拷贝网络副本
    std::shared_ptr<NeuralNetworkProxy> network_copy;
    {
        std::lock_guard<std::mutex> lock(network_mtx);
        network_copy = std::make_shared<NeuralNetworkProxy>(*network_main);
    }

    State s;
    MCTS mcts(2000, 8);

    std::vector<State> states;
    std::vector<std::vector<float>> policies;

    while (!s.is_terminal())
    {
        int action = mcts.search(s, *network_copy); // 每步使用副本
        auto pi = mcts.get_policy_distribution(mcts.root_.get());

        states.push_back(s);
        policies.push_back(pi);

        s = s.next_state(action);
    }

    float result = s.terminal_value();

    for (size_t i = 0; i < states.size(); ++i)
        replay.push(states[i], policies[i], result);
}

int runMcts()
{
    auto network_main = std::make_shared<NeuralNetworkProxy>();
    ReplayBuffer replay(100000);

    const int num_episodes = 1000;

    for (int episode = 0; episode < num_episodes; ++episode)
    {
        // 启动自博弈线程
        std::thread t(self_play_episode, network_main, std::ref(replay));

        // 主线程可以做网络训练（如果有足够数据）
        if (replay.size() > 1024) // 假设最小 batch
        {
            auto batch = replay.sample(512);
            std::lock_guard<std::mutex> lock(network_mtx);
            // train_network(network_main, batch); 训练方法自己实现
        }

        t.join(); // 等自博弈线程完成
    }

    return 0;
}
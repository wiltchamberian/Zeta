#pragma once
#include "Agent.h"
#include <unordered_map>

namespace zeta {


    struct QSetting {
        int max_play_length = 100;
        int episode_num = 1;
        float alpha = 1.0f; //learning Rate
        float gamma = 0.9; //discount factor
        float c_puct = 1.0;

        //dynamic parameter
        //float epsilon = 0.0f;
        float epsilon_init = 0.5f;
        float epsilon_final = 0.05f;
        float decay_rate = 0.002f;
    };

    class QLearning {
    public:
        struct Setting {
            int max_play_length = 100;
            int episode_num = 1;
            float alpha = 1.0f; //learning Rate
            float gamma = 0.9; //discount factor
            float c_puct = 1.0;

            //dynamic parameter
            //float epsilon = 0.0f;
            float epsilon_init = 0.5f;
            float epsilon_final = 0.05f;
            float decay_rate = 0.002f;
        };
        QLearning(){
            std::random_device rd;
            gen.seed(rd());
        }
        void UpdateState(Node* nd);
        int singlePlay(NodePool<Node>* pool, std::unordered_map<uint64_t,Node*>& talbe);
        int train();
        int chooseActionIndexWithPuct(Node* nd, float epsilon);
        int chooseActionIndexByEpsilonPolicy(Node* nd,float epsilon);
        void expand(Node* cur);
        std::shared_ptr<State> play(const std::shared_ptr<State>& state) const;
        Node* createNodeIfNotExist(std::shared_ptr<State> st);
        void Backup(Node* nd);

        int train_mcts();
        int mctsPlay();

        Node* root = nullptr;
        std::mt19937 gen;
        
        std::unordered_map<uint64_t, Node*> table;
        NodePool<Node> pool;
        QSetting setting;
        std::shared_ptr<Proxy> proxy = nullptr;

    };

    
}
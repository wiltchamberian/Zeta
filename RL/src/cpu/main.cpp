#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "NeuralNetwork.h"
#include <thread>
#include <chrono>
#include <type_traits>

#include "test.h"
#include "cu_tool.h"
#include "cnn_test.h"
#include "ThreeTac.h"
#include "Gomoku.h"
#include "mnist.h"
#include "LeNet.h"
#include "Go.h"
#include "vm_test.h"

using namespace zeta;

void GenAllTicTacSamples() {
    TicTacProxy tp;
    TicTac b;
    b.board[0] = 0;
    b.board[1] = 1;
    b.board[2] = 1;
    b.board[3] = -1;
    b.board[4] = 1;
    b.board[5] = 0;
    b.board[6] = -1;
    b.board[7] = 0;
    b.board[8] = -1;
    b.player = 1;

    auto start = std::chrono::high_resolution_clock::now();
    int c = tp.DiscoverAlphaBeta(&b, 10);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << " ms\n";

    std::vector<TicTac> states = tp.ComputeAllStates();
    std::vector<int> values(states.size(), 0);
    for (int i = 0; i < states.size(); ++i) {
        int value = tp.DiscoverAlphaBeta(&states[i], 10);
        tp.pool.Clear();
    }
    //save to file
    BinaryStream bs;
    TicTac::WriteBinary(states, bs);
    bs.saveToFile("tictacs.bin");
}


int main()
{
    //test_vm();

    CudaInit();

    CudaGetDeviceProps();

    //test_cnn_linear();
    //test_dnn_linear();
    //test_cnn_conv();
    //test_dnn_conv();
    //test_cnn_tictac();
    //test_blaslt();
    //mnist_test();//please modify the path for testing!

    BinaryStream bs;
    bs.loadFromFile("tictacs.bin");
    std::vector<TicTac> res = TicTac::ReadBinary(bs);
    std::unordered_map<uint64_t, int> valuesMap;
    for (int i = 0; i < res.size(); ++i) {
        valuesMap[res[i].Hash()] = res[i].value;
    }
    

    //construct optimal policy
    std::vector<Tensor> acts(res.size());
    for (int i = 0; i < res.size(); ++i) {
        acts[i] = Tensor(9);
        auto actions = res[i].legalActions();
        //must lose, so random choose
        if (res[i].value == -1) {
            for (int j = 0; j < actions.size(); ++j) {
                acts[i](actions[j]) = 1.0 / actions.size();
            }
        }
        else {
            std::vector<int> values(actions.size(), 0);
            for (int j = 0; j < actions.size(); ++j) {
                auto st = res[i].NextState(actions[j]);
                values[j] = valuesMap[st.Hash()];
            }
            //choose the smallest value
            int min = 1;
            for (int j = 0; j < values.size(); ++j) {
                min = std::min<int>(values[j], min);
            }
            //find number equal to smalles
            int count = 0;
            for (int j = 0; j < values.size(); ++j) {
                if (values[j] == min) {
                    count += 1;
                }
            }
            //assign probability
            for (int j = 0; j < actions.size(); ++j) {
                if (values[j] == min) {
                    acts[i](actions[j]) = 1.0 / count;
                }
            }
        }
    }
    int batchSize = res.size();
    std::vector<Tensor> trainData(res.size()/batchSize);
    std::vector<Tensor> ys(res.size() / batchSize);
    std::vector<Tensor> vs(res.size() / batchSize);
    for (int i = 0; i < trainData.size(); ++i) {
        trainData[i] = Tensor(batchSize, 2, 3, 3);
        ys[i] = Tensor(batchSize, 9);
        vs[i] = Tensor(batchSize);
        for (int j = 0; j < batchSize; ++j) {
            trainData[i][j].copy(res[i * batchSize + j].Encode());
            ys[i][j].copy(acts[i * batchSize + j]);
            vs[i](j) = res[i * batchSize + j].value;
        }
    }

    //change here to test other proxy, may have runtime error, please report if you find it.
    using ProxyType = TicTacProxy; 


    auto proxy = std::make_shared<ProxyType>();
    proxy->createNNnetwork(0.01f, SGD);
    //proxy->nn->c = 0.0001;

    //we use supervised learning here, only for TicTacProxy
    if constexpr ( false /*std::is_same_v<ProxyType, TicTacProxy>*/ ) {
        int epoch = 20000;
        float lr_min = 0.001f;
        float lr_max = 0.1f;
        float k = 20.0f;
        float loss = 0;
        float loss_prev = 0;
        for (int i = 0; i < epoch; ++i) {
            for (int j = 0; j < trainData.size(); ++j) {
                proxy->train(trainData[j], ys[j], vs[j]);
            }
            float ab = abs(loss - loss_prev);
            float lr = lr_min + (lr_max - lr_min) * expf(-k * ab);
            loss_prev = loss;
            //float lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(3.141592653589 * i / (epoch)));
            if (i < 10000) {
                proxy->setLearningRate(0.1);
            }
            else {
                proxy->setLearningRate(0.02);
            }

        }
    }
    
    mcts::Setting setting;
    setting.simulationCount = 120;
    setting.batchSize = 256;
    setting.miniBatchSize = 256;
    setting.trainStepsPerEpisode = 100;
    setting.num_episodes = 16;
    setting.sample_episodes = 20;
    setting.maxChessLength = 50;
    setting.checkpointCount = 1;
    setting.useDirichletNoise = false;

    setting.targetTemperature = 0.1;
    setting.explorationCount = 10; // minus means not use
    setting.startTemperature = 1000;
    setting.dirichletNoise = 10.0f / proxy->totalActionCount;

    mcts::Mcts mcts;
    mcts.setting = setting;
    mcts.replayBuffer.setMaxSize(5000);

    mcts.mctsProxy = proxy;

    mcts.trainProxy = proxy->Clone();
    mcts.train();
   
    enum FIGHT_TYPE {
        HUMAN_VS_AI,
        AI_VS_AI
    };
    FIGHT_TYPE fightType;
    int d[64];
    bool human = true;
    std::shared_ptr<mcts::State> state = std::make_shared<ProxyType::StateType>();
    state->Init();

    std::cout << "human_vs_ai:0 or ai_vs_ai:1 ?\n";
    std::cin >> d[0];
    if (d[0] == 0) {
        fightType = HUMAN_VS_AI;
    }
    else {
        fightType = AI_VS_AI;
    }
    if (fightType == HUMAN_VS_AI) {
        std::cout << "human first? (1:human 0:AI)\n";
        std::cin >> d[0];
        if (d[0] > 0) {
            std::cout << "huamn first!\n";
            human = true;
        }
        else {
            std::cout << "Ai first!\n";
            human = false;
            state->player = -1;
        }
    }
    else {
        human = true;
    }
    
    state->printState();
    //start play game
    while (true) {
        if (state->is_terminal()) {
            if (state->player == 1) {
                std::cout << "winner is AI!\n";
            }
            else {
                std::cout << "winner is human!\n";
            }
            std::cout << std::endl;
            std::cout << "new game!" << std::endl;
            state->Init();

            std::cout << "human first? (1:human 0:AI)\n";
            std::cin >> d[0];
            if (d[0] > 0) {
                std::cout << "huamn first!\n";
                human = true;
            }
            else {
                std::cout << "Ai first!\n";
                human = false;
                state->player = -1;
            }
            state->printState();
        }
        if (human) {
            if (fightType == HUMAN_VS_AI) {
                std::cout << "please choose your action!\n";
                auto legals = state->legalActions();
                while (true) {
                    std::cin >> d[0];
                    bool bingo = false;
                    for (int k = 0; k < legals.size(); ++k) {
                        if (d[0] == legals[k]) {
                            bingo = true;
                            break;
                        }
                    }
                    if (bingo) {
                        break;
                    }
                }

                state = state->next_state(d[0]);
                state->printState();
                human = !human;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            else if(fightType == AI_VS_AI){
                state = mcts.play(state);
                state->printState();
                human = !human;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            
        }
        else {
            state = mcts.play(state);
            state->printState();
            human = !human;
        }

    }
    std::cin >> d[0];

    return 0;
}

void HumanSupervised() {

}


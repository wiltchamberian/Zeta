#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "NeuralNetwork.h"
#include <thread>
#include <chrono>

//#include <cudnn.h>
#include "test.h"
#include "cu_tool.h"
#include "cnn_test.h"
#include "TicTac.h"

char character(int p) {
    static char chs[3] = { 'O','.' ,'X' };
    return chs[p + 1];
}

void printState(const TicTac& state) {

    std::cout << character(state.board[6]) << " | ";
    std::cout << character(state.board[7]) << " | ";
    std::cout << character(state.board[8]) << std::endl;

    std::cout << character(state.board[3]) << " | ";
    std::cout << character(state.board[4]) << " | ";
    std::cout << character(state.board[5]) << std::endl;

    std::cout << character(state.board[0]) << " | ";
    std::cout << character(state.board[1]) << " | ";
    std::cout << character(state.board[2]) << std::endl;

    if (state.player == 1) {
        std::cout << "player:human" << std::endl;
    }
    else {
        std::cout << "player:AI" << std::endl;
    }

}

int main()
{
    CudaInit();

    CudaGetDeviceProps();

    //test_cnn_linear();
    //test_cnn_conv();
    //test_cnn_tictac();


    TicTacSetting setting;
    setting.simulationCount = 20;
    setting.batchSize = 256;
    setting.miniBatchSize = 256;
    setting.trainStepsPerEpisode = 1000;
    setting.num_episodes = 500;

    std::unique_ptr<TicTacNNProxy> proxy = std::make_unique<TicTacNNProxy>();
    proxy->createNetwork(0.01);

    TicTacMcts mcts;
    mcts.proxy = proxy.get();
    mcts.setting = setting;
    mcts.train();

    int d[64];
    bool humanFirst = false;
    bool human = true;
    
    TicTac state = TicTac::initState();
    std::cout << "human first? (1:human 0:AI)\n";
    std::cin >> d[0];
    if (d[0] > 0) {
        humanFirst = true;
        std::cout << "huamn first!\n";
        human = true;
    }
    else {
        humanFirst = false;
        std::cout << "Ai first!\n";
        human = false;
        state.player = -1;
    }
    printState(state);
    //start play game
    while (true) {
        if (state.is_terminal()) {
            if (state.player == 1) {
                std::cout << "winner is AI!\n";
            }
            else {
                std::cout << "winner is human!\n";
            }
            break;
        }
        if (human) {
            std::cout << "please choose your action!\n";
            std::cin >> d[0];
            state = state.next_state(d[0]);
            printState(state);
            human = !human;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else {
            state = mcts.play(state);
            printState(state);
            human = !human;
        }

    }
    std::cin >> d[0];

    return 0;
}


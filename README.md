# Zeta 🚀

**Zeta** is a deep learning framework created by *WiltChamberlain*, designed to help users understand how deep learning works while providing a lightweight and extensible platform for C++ developers.

Most open-source deep learning libraries are either Python-based 🐍 or extremely large 📦, making it difficult for developers to grasp their internal workings. Moreover, extending these frameworks with low-level features can be challenging in Python. **Zeta** aims to solve these issues by offering a framework that is both readable 📖 and easy to extend 🛠️.

We want **Zeta** to be accessible to everyone 🌎, including users without any AI background. In the future, we plan to add more features, including but not limited to Transformer models 🔄, distributed architectures 🌐, and more.

Another focus of the project is reinforcement learning 🎮 and game AI 🤖. **Zeta** can serve as a platform for experimenting with reinforcement learning ideas, especially in games, making research and experimentation both educational 📚 and fun 🎉.

---

# Features ✨

Currently, **Zeta** supports:

- CPU-based neural networks with forward and backward propagation 🖥️  
- GPU neural networks (CuNN) using custom CUDA kernels ⚡  
- GPU neural networks (DNN) leveraging cuDNN and cuBLAS 🧠  
- A DAG system for automatic differentiation in backward 🔗  
- Monte Carlo Tree Search (MCTS) simulation system 🎲  
- MNIST dataset examples ✏️  
- Tic-Tac-Toe ❌⭕ and small-board Go 🏯 implementations for testing framework capabilities

---

# Achievements 🏆

On the MNIST dataset (60,000 samples), **Zeta** achieves:

- **3x faster training speed than PyTorch** ⚡  
- Comparable accuracy to PyTorch ✅  
- **Unbeatable Tic-Tac-Toe AI player** 👑

This demonstrates that **Zeta** is not only lightweight and extensible but also highly efficient for practical machine learning tasks 💡.

![mnist](./data/performance.jpg)
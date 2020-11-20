# MuZero PyTorch
 Implementation of [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/pdf/1911.08265.pdf) by DeepMind<br> for CartPole-v0 environment.<br><br>
 
 MuZero + naive tree search is working.<br>
 MuZero + monte carlo tree search (MCTS) is now working.<br>
 
 Work in progress ...<br>
 Improvements: more tricks/hacks for better MCTS training. <br>
 
 # MCTS results
 
 ![training_mcts](https://github.com/Hauf3n/MuZero-PyTorch/blob/master/media/training_mcts.png)
 
 # Naive tree search results
 Search in the fully expanded tree at depth n the maximum discounted value (+ discounted rewards).<br>
 Take the action which is the first action from the root to the maximum node.<br>
 
 ![cartpole_naive_tree_search](https://github.com/Hauf3n/MuZero-PyTorch/blob/master/media/cartpole_naive_tree_search.gif)
 ![training_naive_tree_search](https://github.com/Hauf3n/MuZero-PyTorch/blob/master/media/training_naive_tree_search.png)

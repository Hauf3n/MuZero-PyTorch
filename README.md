# MuZero PyTorch
 Implementation of MuZero | CartPole <br><br>
 
 MuZero + naive-tree-search (instead of MCTS) is working.<br>
 Work in progress ...<br><br>
 Current issue:<br> MCTS makes problems. Value and reward prediction seems to work because of decreasing search values at the end of episodes. <br>
 
 # Naive tree search results
 Search in the fully expanded tree at depth n the maximum discounted value (+ discounted rewards).<br>
 Take the action which is the first action from the root to the maximum node.<br>
 
 ![cartpole_naive_tree_search](https://github.com/Hauf3n/MuZero-PyTorch/blob/master/media/cartpole_naive_tree_search.gif)
 ![training_naive_tree_search](https://github.com/Hauf3n/MuZero-PyTorch/blob/master/media/training_naive_tree_search.png)
 
 
 

 

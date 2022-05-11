# CS394R_final: Using Shapley Values & GAIL to Understand Human Behavior

**Credit to: https://github.com/hcnoh/gail-pytorch for GAIL pytorch implementation we sourced for this project

Our code contribution includes changes to the https://github.com/hcnoh/gail-pytorch source code, specifically in models/gail.py and train.py. 
Additionally, player.py is our own constribution which sources and makes changes to gym.utils.play. 

Our changes to gail.py include the ability to source manually provided trajectories along with tracking additional output which needs to be used for LIME and SHAP explanations. Our changes to train.py include adding the ability to send manually provided trajectories to gail.py and all of the LIME and SHAP explanation generation code. Our changes to gym.utils.play included adding manual keyboard mappings for CartPole-v1 along with ability to track and output state and action trajectories. 

How to use: 

Run player.py in order to play CartPole-v1 manually. This will open up the game and allow you to use the 'w' and 'd' keys to push the cart right and left. Once done, press 'esc' key. State and action trajectories for all games played in the session will be exported as .csv files. 

Run train.py to produce explanations for current trajectory stored in .csv files produced by previous step. 


# Adaptive-Neural-Cellular-Automata
 
Repo for the Adaptive-Energy Neural Cellular Automata (AdaptNCA).

PyTorch implementation of AdaptNCA in `lib/EnergyCAModel.py`, and implementation of regular CA in `lib/CAModel.py`.

**Abstract**: Multicellular organisms grow, maintain and regenerate their bodies in a self-organizing manner, through local interactions with neighboring cells and the environment. The recently developed Neural Cellular Automata (NCA) model has shown that this process can be simulated by using a Neural Network to learn local rules that grow a desired pattern. In NCA, each cell has the same constant probability of being updated (fire rate) at every time step. In this paper, we present Adaptive-Energy Neural Cellular Automata model (AdaptNCA) that addresses some limitations of regular NCA. The main change is a cell-wise fire rate, determined by an additional neuron in the network. The fire rates at each step are interpreted as the square root of the cell's energy expenditure, and the model is trained to minimize the total energy expenditure during growth. We show that AdaptNCA can have better persistence and damage regeneration capabilities than an equivalent NCA model, reaching reconstruction losses up to 46% and 69% lower, respectively.


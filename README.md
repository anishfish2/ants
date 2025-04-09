# Ant Colony Simulation

This project implements various ant colony simulation environments using NEAT (NeuroEvolution of Augmenting Topologies) for training neural networks. The simulations include predator-prey interactions, foraging behaviors, and navigation tasks.

## Project Structure

```
ants/
├── configs/                 # Configuration files for NEAT
├── output/                 # Output files and logs
├── src/                    # Source code
│   ├── models/            # Saved model files
│   ├── videos/            # Generated simulation videos
│   ├── ant.py             # Base ant class and agent implementations
│   ├── food.py            # Food object implementation
│   ├── predator_prey_retrieval_env.py  # Environment for predator-prey with retrieval
│   ├── predator_prey_retrieval_main.py # Main script for predator-prey training
│   └── ...                # Other environment and main files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Simulations

To run the predator-prey retrieval simulation:
```bash
python src/predator_prey_retrieval_main.py
```

## Configuration

The NEAT configuration files are located in the `configs/` directory. These files control the neural network architecture and evolution parameters.

## Output

- Training progress and fitness graphs are saved in the `output/` directory
- Simulation videos are saved in `src/videos/`
- Trained models are saved in `src/models/`

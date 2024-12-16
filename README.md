# Optimization-of-Protein-Ligand-Molecular-Docking-using-AI-ML

## Introduction
Proposed a GNN based framework to predict the binding affinity and best pose evaluation.
Molecular docking is an in-silico method widely utilized in early-stage drug discovery for:
Screening promising drug candidates.
Exploring potential side effects or toxicities.

- Traditionally, tools like AutoDock4 (AD4) and AutoDock Vina (Vina) estimate protein-ligand binding affinities using heuristic scoring functions, balancing computational efficiency and accuracy.

## Challenges
However, these methods face several challenges:
-  Limited ability to capture nuanced molecular interactions.
-  Rigid receptor assumptions.
-  naccuracies in pose prediction for complex or flexible systems.

These limitations result in inefficiencies, where only 20-30% of compounds identified through molecular docking show activity in biological assays. While AD4 and Vina remain valuable for initial screenings, their results often require refinement to align with experimental outcomes.

### Proposed Framework
To address these challenges, we propose a novel framework integrating Graph Convolutional Neural Networks (GCNs) with traditional docking software.

## Key Features
Graph-based Modeling:
- GCNs model protein-ligand complexes as graph data, capturing intricate molecular interactions.
  
Enhanced Prediction:
- Improved accuracy in predicting binding affinities and identifying favorable configurations.
  
Hybrid Approach:
- Combines traditional tools (AD4, Vina) with GCN outputs to optimize docking results.

This integrated method leverages machine learning to refine molecular docking patterns, enhancing both predictive accuracy and computational efficiency.


## Potential Impact 
This approach holds significant potential to:
- Revolutionize computational drug discovery.
- Reduce the development time and costs of new therapeutics.
- Enhance the overall reliability of protein-ligand docking predictions.

### The dataset used for this framework can be accessed at:
https://drive.google.com/drive/folders/1lpUPJIp0Xa7RU-jc7F5AvnAt8D64qt73 


# Optimization-of-Protein-Ligand-Molecular-Docking-using-AI-ML
Optimization of Protein-Ligand Molecular Docking using  Graph Convolutional Neural Networks:  Predicting Binding Affinity and posing
Molecular docking is an in-silico method widely utilized in early-stage drug discovery for screening promising drug candidates and 
exploring potential side effects or toxicities. Traditionally, tools like AutoDock4 (AD4) and AutoDock Vina (Vina) estimate
protein-ligand binding affinities using heuristic scoring functions, providing a balance between computational efficiency and accuracy.
However, these methods face challenges such as limited ability to capture nuanced molecular interactions, 
rigid receptor assumptions, and inaccuracies in pose prediction, particularly in complex or flexible systems. 
These limitations significantly contribute to the inefficiencies observed in drug discovery, 
where only 20-30% of compounds identified through molecular docking show activity in biological assays.
While AD4 and Vina remain valuable for initial screenings, their results often require refinement to align with experimental outcomes.

Dataset used: https://drive.google.com/drive/folders/1lpUPJIp0Xa7RU-jc7F5AvnAt8D64qt73 

To address and overcome these challenges, we propose a novel framework integrating Graph Convolutional Neural Networks (GCNs) with traditional docking software.
GCNs excel in modeling protein-ligand complexes as graph-based data, enabling more precise predictions of binding affinities and identifying the most 
energetically favorable configurations. This method enhances the predictive accuracy and computational efficiency of docking processes. 
By combining the output of traditional tools like AD4 and Vina with the predictive capabilities of GCNs, 
our framework aims to optimize binding affinity prediction and pose identification.
This integrated approach leverages machine learning to learn from molecular docking patterns, 
significantly enhancing both accuracy and efficiency in computational drug discovery, potentially reducing the time and costs associated with developing new therapeutics.


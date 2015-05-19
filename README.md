# master_thesis
Optimizing Support Vector Machines with Characteristic Boundary Points

Abstract:

In this master thesis two supervised learning classification methods are studied. The research is aimed to study the 
Characteristic Boundary Points from the Optimized Geometric Ensembles (OGE) methodology in order to include them in 
the Support Vector Machines formulation. The new algorithm is proposed in batch and on-line fashions and the difference 
between both formulations are studied. With this exploration, a gain of information from the optimal boundary is obtained 
in the reformulated SVMs, as a consequence of the introduction of the CBP, that may lead to the optimization of the tuning
stage. In the online case, an automation of the parameter from the RBF Kernel improves the hyperparameter tuning time 
complexity. The results are promising for this new algorithm that combines both methodologies and prevents the overfitting
problem. A further line of research could be followed combining the Characteristic Boundary Points with different types of
Kernel as well as with other classification algorithms.

Packages:
- cvx : to run the code the convex optimization package is needed

To run:
- mainBATCH  : runs the new methodology SVM-CBP in batch fashion, obtaining results for SVM-CBP and SVMs 
- mainONLINE : runs the new methodology SVM-CBP in online fashion, obtaining results for SVM-CBP and SVMs  

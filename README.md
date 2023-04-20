# Imbalanced_Classification_for_Resonant_Dynamics
A repository for software and data bases created for analyzing the effect of using Imbalanced classification for problems in asteroid resonant dynamics.

Machine learning (ML) applications for studying asteroid resonant dynamics are a relatively new field of study.  Results of several different approaches are currently available for asteroids interacting with the $z_2$, $z_1$, M1:2, and ${\nu}_6$ resonances. However, one challenge when using ML to the databases produced by these studies is that there is often a severe imbalance ratio between the number of asteroids in librating orbits and the rest of the asteroidal population. In some cases, the ratio between the two classes can be as high as 1:270. This class imbalance can impact the performance of classical ML algorithms, that were not designed for such severe imbalances. Various techniques have been recently developed to address this problem, including cost-sensitive strategies, methods that oversample the minority class, undersample the majority one, or combinations of both. Here, we investigate the most effective approached for improving the performance of ML algorithms for known resonant asteroidal databases.

We are providing the dual classes data bases for the studied resonances in the DATA branch, and all the code applying Imbalanced Classification methods for the aligned case of the ${\nu}_6$ resonance in the MODELS branch.  The codes can be easily adapted for the other data bases.

More information on this work can be obtained at Carruba et al. (2023) Imbalanced classification applied to asteroid
resonant dynamics, currently under review at Frontiers in Astronomy and Space Sciences.

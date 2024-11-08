# Autonomous_system_Project3
This project focuses on three topics: Particle Filter, Iterative Closest Points (ICP), and Visual Odometry. Each topic involves practical implementations and analysis of results. 

 

Particle Filter: 

The Particle Filter section introduces the concept of sampling a distribution using particles, generating predictions based on motion models, and updating the predictions using measurements and their uncertainties. The implementation involves loading Odometry and Ground Truth (GT) data, plotting the GT trajectory and landmarks, and utilizing a spin LiDAR 2D sensor for measurement. The Particle Filter algorithm is then executed with the noised GT samples The results are analysed by comparing them to the GT trajectory, calculating Mean Square Error (MSE), and determining the minimal set of particles that maintain good performance. The advantages and disadvantages of Particle Filters are discussed, along with suggestions for overcoming the limitations. 

 

ICP - Iterative Closest Points: 

The ICP section focuses on implementing the Iterative Closest Point algorithm on two different scan frames under varying conditions. The vanilla ICP algorithm is implemented using a KDTree for data association. Three different mechanisms and conditions are tested: full point cloud with KDTree associations, filtered point cloud with KDTree associations, and filtered point cloud with Nearest Neighbors associations. The results are analyzed by comparing the point clouds before and after scan correction, examining convergence time, error, and the performance of different association methods. The success and failure of the scan correction process are analyzed, along with potential reasons for mismatched objects. 

 

Visual Odometry: 

The Visual Odometry section aims to implement a monocular visual odometry pipeline using essential features such key point tracking, pose estimation, and refining of rotation and translation. The dataset used is the KITTI visual odometry dataset, and the algorithm is calibrated to ensure a maximum Euclidean distance of 15 meters from the ground truth within the first 500 frames. The algorithm pipeline is described, and the results are analysed through animations, comparison with the ground truth trajectory, and identifying reasons for drift.  

Overall, this assignment provides hands-on experience with Particle Filters, ICP, and Visual Odometry, along with in-depth analysis and suggestions for improvement. The practical implementations are carried out using a Google Colab notebook for Particle Filter and ICP, and a python script on the KITTI visual odometry dataset for Visual Odometry. 

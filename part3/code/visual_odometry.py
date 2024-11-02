import numpy as np
import cv2
from data_loader import DataLoader
from camera import Camera
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import imageio

basedir = os.path.dirname(os.path.realpath(__file__))
class VisualOdometry:
    def __init__(self, vo_data):
        """
        Initialize the VO class with the loaded data vo_data
        lastly, initialize the neutral rotation and translation matrices
        """
        self.vo_data = vo_data

        # initial camera pose
        self.camera_rotation = np.eye(3) #vo_data.cam.extrinsics[:, :3] #TODO
        
        self.camera_translation = np.zeros((3,1)) #TODO
        
    def calc_trajectory(self):
        """
        apply the visual odometry algorithm
        """
        gt_trajectory = np.array([]).reshape(0, 2)
        measured_trajectory = np.array([]).reshape(0, 2)
        key_points_history = []
        frames =[]
        xz_error_arr = np.zeros(self.vo_data.N) #TODO

        prev_img = None
        prev_gt_pose = None
        i = 0
        test_traj = np.array([]).reshape(0, 2)
        fig = plt.figure(figsize=[16, 12])
        ax_image = fig.add_subplot(1, 3, 1)
        ax_error_plot = fig.add_subplot(1, 3, 2)
        ax_trajectory = fig.add_subplot(1, 3, 3)
        for curr_img, curr_gt_pose in zip(self.vo_data.images, self.vo_data.gt_poses):
            if prev_img is None:
                prev_img = curr_img
                prev_gt_pose = curr_gt_pose
                continue
            
             # ***********
            # TODO
            # ***********
            #first we need to extract key features from and dscription from both frames
            detector = cv2.SIFT_create()
            prev_keypoints, prev_descriptor = detector.detectAndCompute(prev_img, None)
            curr_keypoints, curr_descriptor = detector.detectAndCompute(curr_img, None)
            
            key_points_history.append(prev_keypoints)
            
            # now we will match the features between images
            bf = cv2.BFMatcher()
            match_list = bf.knnMatch(prev_descriptor,curr_descriptor,k=2)#.match(prev_descriptor,curr_descriptor)#
            
            # Apply ratio test
            matches = []
            for m,n in match_list:
              #print("m.distance", m.distance, "n.distance",n.distance)
              if m.distance < 0.6*n.distance:
                  matches.append(m)
            
            # now we need to find the rotation matrix and transalation matrix between 2 frames
            prev_points = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]) #np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)#
            curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]) #np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)#
            
            #we can find the essential matrix 
            essential_mat,_ = cv2.findEssentialMat(curr_points,                          
                           prev_points,
                          self.vo_data.cam.intrinsics,method=cv2.RANSAC,prob=0.99,threshold=1.0)
            #This function decomposes an essential matrix using decomposeEssentialMat and then verifies possible pose hypotheses by doing cheirality check. 
            #The cheirality check means that the triangulated 3D points should have positive depth:
            _ , rotation_mat , translation_vector, mask = cv2.recoverPose(essential_mat,
                                                                      curr_points,
                                                                      prev_points,
                                                                      self.vo_data.cam.intrinsics)
         
            
            scale = np.sqrt(((curr_gt_pose[:, 3] - prev_gt_pose[:, 3]) ** 2).sum()) / np.sqrt(
                (translation_vector ** 2).sum())
            
            
            self.camera_translation +=   scale* self.camera_rotation @ translation_vector
            self.camera_rotation = self.camera_rotation @ rotation_mat

            gt_trajectory = np.concatenate((gt_trajectory, np.array([[curr_gt_pose[0, 3]
                    , curr_gt_pose[2, 3]]])), axis=0) # groundTruth
            measured_trajectory = np.concatenate((measured_trajectory,
                np.array([[float(self.camera_translation[0]),
                float(self.camera_translation[2])]])), axis=0)
                        
             
            xz_error_arr = np.linalg.norm((gt_trajectory.reshape(-1, 2) - measured_trajectory.reshape(-1, 2)), axis=1)
            
            # since there are so many frames and my computer has not much memory left
            # we will save the first 500 frames each an then every 4th from the trajcrtory
            if i<= 500 or i %4 ==0 or i == self.vo_data.N:
                fig2 = visualization(GT_location = gt_trajectory,VO_location=measured_trajectory,prev_points=prev_points,curr_points=curr_points,xz_error_arr = xz_error_arr
                        ,curr_image = curr_img, ax_image = ax_image,ax_error_plot =ax_error_plot, ax_trajectory =ax_trajectory, frame_idx = i, idx_frames_for_save = [i], dest_dir ='Results')
                canvas = fig2.canvas
                canvas.draw()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(canvas.get_width_height()[::-1] + (3,))
                frames.append(image)  
                plt.close('all')
            prev_img = curr_img
            prev_gt_pose = curr_gt_pose
            i+=1
            
            
               
                
            
            
        print("Saving animation")
        
        imageio.mimsave( './VO_Tomer.mp4',frames)
        return gt_trajectory, measured_trajectory, key_points_history

 
 
# Helpfull reference
# @staticmethod
def visualization(GT_location,VO_location,prev_points,curr_points,xz_error_arr,curr_image,ax_image, ax_error_plot, ax_trajectory, frame_idx,idx_frames_for_save,dest_dir):
     """
     plot the graphes of the VO include: image, GT and estimated trajectory, features.
     :param GT_location: GT location
     :param VO_location: VO estimated location
     :param prev_points: KeyPoints from the previous frame.
     :param curr_points: match KeyPoints in the current frame of the previos frame.
     :param xz_error_arr: euclidian distance error in (x,y)
     :param curr_image: current image
     :param ax_image: Axis object for the image
     :param ax_error_plot: Axis object for the error plot
     :param ax_trajectory: Axis object for the trajectory plot
     :param frame_idx: frame index
     :param idx_frames_for_save: the indexes of the frames we want to save their graphs.
     :param dest_dir: the directory name for saving the graphs and animations to.
     :return: the frame graph.
     """
     Frame=[]
     plot_0=ax_image.imshow(curr_image,cmap='gray')
     Frame.append(plot_0)
     plot_1=ax_image.scatter(curr_points[:,0],curr_points[:,1],s=2,linewidths=0.5,edgecolors="b",marker="o")
     Frame.append(plot_1)
     plot_2=ax_image.scatter(prev_points[:, 0], prev_points[:, 1], s=2,linewidths=0.5, edgecolors="g",marker="P")
     Frame.append(plot_2)
     plot_3,=ax_trajectory.plot(VO_location[:,0],VO_location[:,1],c="r")
     Frame.append(plot_3)
     plot_4,=ax_trajectory.plot(GT_location[:, 0], GT_location[:, 1],"--b")
     Frame.append(plot_4)
     plot_5, = ax_error_plot.plot(xz_error_arr,c="orange")
     Frame.append(plot_5)
     if frame_idx == 1:
         ax_image.legend(["current key points", "Previous key points"],loc="upper right")
         ax_trajectory.legend(["VO-Estimated with scale", "GT"],loc="upper right")
         ax_trajectory.grid()
         ax_error_plot.grid()
     if frame_idx in idx_frames_for_save:
         fig_2 = plt.figure(figsize=[16, 12])
         grid = plt.GridSpec(12, 17, hspace=0.2, wspace=0.2)
         ax_image_2 = fig_2.add_subplot(grid[:5, :], title="Scene Image,Frame: {}".format(frame_idx))
         ax_error_plot_2 = fig_2.add_subplot(grid[6:, :8], title="Euclidean Distance Error", xlabel="Frame number",ylabel="Error[m]")
         ax_trajectory_2 = fig_2.add_subplot(grid[6:, 9:], title="Trajectory", xlabel="X[m]", ylabel="Y[m]",xlim=(-50, 750), ylim=(-100, 1000))
         ax_image_2.axis('off')
         ax_image_2.imshow(curr_image, cmap='gray')
         ax_image_2.scatter(curr_points[:, 0], curr_points[:, 1], s=2, linewidths=0.5, edgecolors="b", marker="o")
         ax_image_2.scatter(prev_points[:, 0], prev_points[:, 1], s=2, linewidths=0.5, edgecolors="g", marker="P")
         ax_trajectory_2.plot(VO_location[:, 0], VO_location[:, 1], c="r")
         ax_trajectory_2.plot(GT_location[:, 0], GT_location[:, 1], "--b")
         ax_error_plot_2.plot(xz_error_arr, c="orange")
         ax_image_2.legend(["current key points", "Previous key points"], loc="upper right")
         ax_trajectory_2.legend(["VO-Estimated with scale", "GT"], loc="upper right")
         ax_trajectory_2.grid()
         ax_error_plot_2.grid()
         if not os.path.exists(dest_dir):
             os.mkdir(dest_dir)
         plt.savefig(dest_dir + "/Visual Odometry Frame #{}".format(frame_idx))
         
     return plt.gcf()
# @staticmethod
def save_animation(ani, basedir, file_name):
    """
    save animation function
    :param ani: animation object
    :param basedir: the parent dir of the animation dir.
    :param file_name: the animation name
    :return: None
    """
    print("Saving animation")
    if not os.path.exists(basedir + "/Animation videos"):
        os.makedirs(basedir + "/Animation videos")
    gif_file_path = os.path.join(basedir + "/Animation videos", f'{file_name}.gif')
    mp4_file_path = os.path.join(basedir + "/Animation videos", f'{file_name}.mp4')
    writergif = animation.PillowWriter(fps=30)
    ani.save(gif_file_path, writer=writergif)
    clip = mp.VideoFileClip(gif_file_path)
    clip.write_videofile(mp4_file_path)
    os.remove(gif_file_path)
    print("Animation saved")
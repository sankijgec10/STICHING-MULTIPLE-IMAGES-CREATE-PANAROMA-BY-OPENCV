# STICHING-MULTIPLE-IMAGES-CREATE-PANAROMA-BY-OPENCV
Panaroma is nothing but a wide-angle representation of a scene and it is usually built from multiple images captured at a single location with slightly different viewpoints.  Here we stiches multiple image to form a single image by merging the common part of this image .We warp images based on the matching keypoints .In this technique we used the ORB detector to extract the keypoints. Once we extracted the keypoints  we used the Brute Force matcher to match the  descriptors. Brute Force matching is pretty straightforward. For every descriptor in the first image, We match it with every descriptor in the second image and take the closest one. To compute the closest descriptor we use the Hamming distance as the metric. 

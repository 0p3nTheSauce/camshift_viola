# camshift_viola
Camshift and Viola jones for the third prac of image processing

# Objectives
Get CamShift working and use it to track an object/face. 

Get Viola Jones working and use it to track an object/face.

Use Viola Jones eye detector to find a skin sample in the nose region and apply it to CamShift to do skin segmentation.

You may use code from the internet and openCV for this.

# Outcomes:
in smart_segmentation.py, viola jones is used to get a skin sample between the eyes, which ca be seen as the red square. A histogram is generated from this square area of pixels. The histogram is used by camshift for tracking. Additionaly the histogram is used to create an upper, and lower bound hue. A binary segmentation mask is created from the upper and lower bound hue, and applied to the video feed for skin segmentation
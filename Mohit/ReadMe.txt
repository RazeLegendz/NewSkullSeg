Inputs (Mat files, Seg Files and CT Files):  https://drive.google.com/drive/u/1/folders/1w18aKJ9JDHNhLgaYbv2kTMFMhnUV5KW2

Reference Paper Shared by Dr. Kim Jinkoo at start of project:
https://www.sciencedirect.com/science/article/pii/S1053811916000306 

Reference Word File shared by Dr. Kim Jinkoo at start of project attached under name of "Introduction.doc"

Actually I started late on project, I was doing other project , as per my last conversation with Darren.
 
He told,Dr. Kim Jinkoo taught him everything relating to how to best perform segmentation in Seg3d2 Tool and can most likely explain and teach the material better than he himself can).

So for manually skull annotation, we have to basically meet Dr. Kim Jinkoo.

My part :
Generally I tried to implement CNN U-NET model for segmentation, and tried to run over GPU cluster in batches. 1 input image per batch due to memory constraints on GPU. 
Due to time constraints, I was not able to output segmented image, but in my tensorflow code I have written method which compares pixel by pixel with hand annotated image.

I have uploaded my code for reference ("Unet.py").

Initially due to various problem in gpu and tensorflow session graph limit size of 2 GB, I was not able to run UNet.py, so I modified my CNN architecture(basically I reduced no. 
of filters and no of inputs in training) to atleast run my code and check.

I have uploaded my code for reference ("Test.py").

Both are well commented code.
 
Paper Referred by me:  
1) https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdfv

I am planning to implement following Github link for my next semester:
Deep learning based skull stripping and FLAIR abnormality segmentation in brain MRI using U-Net architecture
1) https://github.com/mateuszbuda/brain-segmentation



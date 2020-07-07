# 0.91 run

1. <em>preprocess_iafoss.py</em> I preprocess tiff images into pickle files each containing a set of tiles from one whole slide image of shape 36x256x256x3
2. <em> downsample.py</em> Downsample to 36x128x128x3. I use these images for the first 10 epochs, and then switch to full size images)

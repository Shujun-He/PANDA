# 0.91 run

1. <em>preprocess_iafoss.py</em> I preprocess tiff images into pickle files each containing a set of tiles from one whole slide image of shape 36x256x256x3
2. <em> downsample.py</em> Downsample to 36x128x128x3 and save to a different dir. I use these images for the first 10 epochs, and then switch to full size images)
3. <em>train_4_fold.py</em> to train. In my 0.91 i used fold 2 as the validation set, so here it only uses fold 2 to train.
4. After training, use <em>get_best_weights.py</em> to get the best weights for submission, usually I use top 3 checkpoints. My 0.91 run was selected based on accuracy instead of qwk

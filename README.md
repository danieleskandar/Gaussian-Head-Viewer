## Dynamic Gaussian Visualizer

This is a simple Dynamic Gaussian Splatting Visualizer specialized in head with hair models built with PyOpenGL. It's easy to install with minimum dependencies. The goal of this project is to provide a visualizer focused on head models with hair strands as those in *(paper).* 
![Banner image](videos/banner.gif)


## Usage

Clone the repository:

```
git clone https://github.com/Daniel-Eskandar/Dynamic-Gaussian-Visualizer.git
```

Install the dependencies:

```
pip install -r requirements.txt
```

Launch the viewer and check how to use UI in the "help" panel.

```
python main.py
```

Download models.zip folder from this [Google Drive link](https://drive.google.com/drive/folders/.1ExPZ1vI3E5ZMtiK0IlkiZ2bG6fu5VnoS)


The Gaussian file loader is compatible with the official implementation but works its best with models as in *(paper)*. For files with hair strands it is necessary for these gaussians’ means, scales and rotations to be the first rows in the ply file. Once in this order, the following command can be run to save the number of hair strands, `nhairs`, and number of gaussians per strand, `ngauss_strand` to the ply file:

```
python utils/util.py my_path --n_strands=12000 --n_gaussians_per_strand=31
```

For the curls feature to load the rotation matrices instead of computing them on the fly, the following command can be run to pre-compute these where `nsamples` is the number of evenly spaced values for both amplitude and frequency, and `max_amp` and `max_freq` :

```
python utils/frenet_arcle.py my_path n_samples=0 --max_amp=0.025 --max_freq=3
```

Finally, for the frames to be loaded as a single numpy array and getting a speedup by reducing the number of read operations the following python script takes the directory in which the files `frame_#_mean_frenet.npy`, `frame_#_rot_frenet.npy` , and `frame_#_scale_frenet.npy` and whether the rotation is represented as a rotation matrix or a quaternion:

```
python utils/frame_packer.py my+path --rot_format {quat, mat}
```

### Features

1. Hair coloring
<p align="center">
  <img src="videos/coloring.gif" width="70%" height="70%"/>
</p>

2. Hair cutting
<p align="center">
  <img src="videos/cutting.gif" width="70%" height="70%"/>
</p>

3. Export ply file with edited color and cut

4. Curly hair effect 
<p align="center">
  <img src="videos/curls.gif" width="70%" height="70%"/>
</p>

5. Dynamical gaussians for hair dynamics
<p align="center">
  <img src="videos/frames.gif" width="70%" height="70%"/>
</p>

6. Axes view as renderer
<div align="center">
  <img src="/videos/axes.gif" width="40%" height="40%"/>
</div>

7. Display multiple heads
8. Show or hide hair and head

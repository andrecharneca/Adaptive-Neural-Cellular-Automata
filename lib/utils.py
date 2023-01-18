
from __future__ import print_function
import sys
import numpy as np
import imageio
from lib.utils_vis import to_rgb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from IPython.display import HTML
import GPUtil


def tup_distance(node1, node2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return ((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)**0.5
    elif mode=="Manhattan":
        return np.abs(node1[0]-node2[0])+np.abs(node1[1]-node2[1])
    else:
        raise ValueError("Unrecognized distance mode: "+mode)


def mat_distance(mat1, mat2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return np.sum((mat1-mat2)**2, axis=-1)**0.5
    elif mode=="Manhattan":
        return np.sum(np.abs(mat1-mat2), axis=-1)
    else:
        raise ValueError("Unrecognized distance mode: "+mode)


def debug(*args, same_line=False):
    ''' Pass as strings, prints expression name and value '''
    frame = sys._getframe(1)
    
    for var in args:
      if same_line:
        print(var, '=', repr(eval(var, frame.f_globals, frame.f_locals)), end='; ')
      else:
        print(var, '=', repr(eval(var, frame.f_globals, frame.f_locals)))
    #if same_line: print('\n')
    

def load_emoji(index, path="data/emoji.png"):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji
  
def load_lizard(path="data/lizard_clean.png"):
    im = imageio.imread(path)
    im = np.array(im.astype(np.float32))
    im /= 255.0
    return im

def visualize_batch_ponder(x0, x, n_steps):
  '''
    View batch initial seed, state at halting and final state
  '''
  vis0 = to_rgb(x0)
  vis_final = to_rgb(x[-1])
  print('batch (before/after):')
  plt.figure(figsize=[15,5])
  for i in range(x0.shape[0]):
      plt.subplot(3,x0.shape[0],i+1)
      plt.imshow(vis0[i])
      plt.axis('off')
  for i in range(x0.shape[0]):
      vis_halt = to_rgb(x[n_steps[i]-1, i, ...])
      plt.subplot(3,x0.shape[0],i+1+x0.shape[0])
      plt.imshow(vis_halt)
      plt.text(0,0, "n_halt="+str(n_steps[i]))
      plt.axis('off')
  for i in range(x0.shape[0]):
      plt.subplot(3,x0.shape[0],i+1+2*x0.shape[0])
      plt.imshow(vis_final[i])
      plt.text(0,0, "n_final="+str(x.shape[0]))
      plt.axis('off')

  plt.show()


def visualize_batch_adapt(x0, x, lambdas_steps, update_grid_steps=None, progress_steps = 8):
    '''
      View batch initial seed, batch final state, and progression of lambdas
      and update grid steps of the 1st batch item
    '''
    vis_final = to_rgb(x[-1])
    vis_batch0 = to_rgb(x[:,0])
    max_steps = x.shape[0]
    print('batch (before/after):')
    plt.figure(figsize=[25,12])
    n_cols = max(x0.shape[0], progress_steps)
    n_rows = 5

    # final states
    for i in range(vis_final.shape[0]):
      plt.subplot(n_rows,n_cols,i+1)
      plt.imshow(vis_final[i])
      plt.text(0,0, "n_final="+str(max_steps))
      plt.axis('off')

    # visualize evolution of first in batch
    for i in range(progress_steps):
      plt.subplot(n_rows, n_cols, i+1+n_cols)
      plt.imshow(vis_batch0[max_steps//progress_steps * (i+1)-1])
      plt.text(0,0, "step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')

    # visualize lambdas
    for i in range(progress_steps):
      plt.subplot(n_rows, n_cols, i+1+2*n_cols)
      plt.imshow(lambdas_steps[max_steps//progress_steps * (i+1)-1,0])
      plt.text(0,0, "lambdas at step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')
    plt.colorbar()

    # visualize update_grid
    for i in range(progress_steps):
        plt.subplot(n_rows, n_cols, i+1+3*n_cols)
        plt.imshow(update_grid_steps[max_steps//progress_steps * (i+1)-1,0])
        plt.text(0,0, "updates at step="+str(max_steps//progress_steps * (i+1)-1))
        plt.axis('off')
    plt.show()


def visualize_batch_energy(x0, x, fireRates_steps, update_grid_steps=None, progress_steps = 8):
    '''
      View batch initial seed, batch final state, and progression of fireRates
    of the 1st batch item
    '''
    vis_final = to_rgb(x[-1])
    vis_batch0 = to_rgb(x[:,0])
    max_steps = x.shape[0]
    print('batch (before/after):')
    plt.figure(figsize=[25,12])
    n_cols = max(x0.shape[0], progress_steps)
    n_rows = 5

    # final states
    for i in range(vis_final.shape[0]):
      plt.subplot(n_rows,n_cols,i+1)
      plt.imshow(vis_final[i])
      plt.text(0,0, "n_final="+str(max_steps))
      plt.axis('off')

    # visualize evolution of first in batch
    for i in range(progress_steps):
      plt.subplot(n_rows, n_cols, i+1+n_cols)
      plt.imshow(vis_batch0[max_steps//progress_steps * (i+1)-1])
      plt.text(0,0, "step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')

    # visualize fireRates
    for i in range(progress_steps):
      plt.subplot(n_rows, n_cols, i+1+2*n_cols)
      plt.imshow(fireRates_steps[max_steps//progress_steps * (i+1)-1,0])
      plt.text(0,0, "fireRates step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')
      plt.colorbar()
    plt.show()


def visualize_batch(x0, x):
    vis0 = to_rgb(x0)
    vis_final = to_rgb(x)
    print('batch (before/after):')
    plt.figure(figsize=[15,5])
    for i in range(x0.shape[0]):
        plt.subplot(3,x0.shape[0],i+1)
        plt.imshow(vis0[i])
        plt.axis('off')
    for i in range(x0.shape[0]):
        plt.subplot(3,x0.shape[0],i+1+x0.shape[0])
        plt.imshow(vis_final[i])
        plt.axis('off')

    plt.show()

def plot_loss_reg_rec(loss_log, loss_reg_log, loss_rec_log):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=[25,5],nrows=1, ncols=3, sharey=True)
  
    ax1.set_title('total loss (log10)')
    ax1.plot(np.log10(loss_log), '.', alpha=0.2, label='total')
    
    ax2.set_title('beta*regularization loss (log10)')
    ax2.plot(np.log10(loss_reg_log), 'g.', alpha=0.2, label='beta*regularization')

    ax3.set_title('reconstruction loss (log10)')
    ax3.plot(np.log10(loss_rec_log), 'r.', alpha=0.2, label='reconstruction')
    plt.show()

def plot_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.2, label='total')
    plt.legend()
    plt.show()

def plot_avg_steps(avg_steps_log):
    plt.figure(figsize=(10, 4))
    plt.title('Avg steps history')
    plt.plot(avg_steps_log, 'g.', alpha=0.1)
    plt.show()

def animate_steps(*arr_steps, colorbar_plots=None, interval=40, titles = None, num_cols=None):
  ''' 
  Input: arrays to animate. List of indexes of the plots that need colorbars. It'll only show the first batch item
  arr_steps[i].shape = (max_steps, batch_size, img_size, img_size) 
  colorbar_plots = list of indexes of the plots with colorbars (assuming values go between 0 and 1)
  '''
  max_steps = arr_steps[0].shape[0]
  num_plots = len(arr_steps)
  num_cols = num_plots if num_cols is None else num_cols
  num_rows = num_plots//num_cols+1 if num_plots%num_cols!=0 else num_plots//num_cols
  fig, axs = plt.subplots(num_rows, num_cols)
  axs = axs.flatten() if num_rows>1 else axs
  frames = []
  if num_plots==1: axs = [axs]
  if titles is None: 
    titles = [""]*len(axs)
  else:
    # pad titles to len(axs)
    titles = titles + [""]*(len(axs)-len(titles))
  for i in range(len(axs)):
      axs[i].set_title(titles[i])
      axs[i].get_xaxis().set_visible(False)
      axs[i].get_yaxis().set_visible(False)

  for i in range(num_plots):
    axs[i].imshow(arr_steps[i][0,0])

  for k in range(max_steps):
    ims = []
    ttl = plt.text(0.5, -0.1, f"Step {k}", horizontalalignment='center', verticalalignment='bottom', transform=axs[0].transAxes)
    for i in range(num_plots):
      ims.append(axs[i].imshow(arr_steps[i][k,0], animated=True, vmin=0, vmax=1, cmap='jet'))
    ims.append(ttl)  
    frames.append(ims)

  # colorbars
  if colorbar_plots is not None:
    for i in colorbar_plots:
      fig.colorbar(ims[i], ax=axs[i],fraction=0.046, pad=0.04)
  
  anim = ArtistAnimation(fig, frames, interval=interval, blit=True)
  plt.close()
  return HTML(anim.to_html5_video())


def get_free_gpu(min_free_mem=0.9):
    # get the gpu with the most free memory
    print("Getting free GPU...")
    GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()

    # return first GPU with more than 90% free memory
    for gpu in GPUs:
        if gpu.memoryFree > min_free_mem * gpu.memoryTotal:
            print("Using GPU: ", gpu.id)
            return gpu.id
    else:
      # throw error
      raise Exception(f"No GPU with more than {min_free_mem*100}% free memory found")
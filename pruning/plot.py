"""Generate nice plots for paper."""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import wandb
api = wandb.Api()

from itertools import cycle

MARKERSIZE = 10
LINEWIDTH = 4

# ============================================================================
# Utils
# ============================================================================

def smooth_data(scalars, weight=0.):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def tsplot(data, x=None, smooth=0., marker=None, label=None, **kw):
    if x is None:
        x = np.arange(data.shape[0])
    # Plot data's smoothed mean
    y = np.mean(data, axis=1)
    y = smooth_data(y, weight=smooth)
    # Find standard deviation and error
    sd = np.std(data, axis=1)
    se = sd/np.sqrt(data.shape[1])
    # Plot
    plt.plot(x, y, marker=marker, markersize=MARKERSIZE, linewidth=LINEWIDTH, label=label, **kw)
    # Show error on graph
    cis = (y-se, y+se)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)

def plot_legend(legends, colors, markers, save_name):
    # Dummy plots
    for legend, color, marker in zip(legends, colors, markers):
        plt.plot([0,0,0], [0,0,0], color=color, label=legend, marker=marker, markersize=MARKERSIZE, linewidth=LINEWIDTH)
    # Get legend separately
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(handles, labels, loc='center', ncol=len(legends))
    plt.axis('off')
    fig = leg.figure
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_name, bbox_inches=bbox, pad_inches=0, dpi=500)
    plt.close('all')

# ============================================================================
# Main plotting
# ============================================================================

def retrieve_group(project, group, metric, x_axis, prepend=None):
    # Get runs
    path = os.path.join(entity, project)
    runs = api.runs(path=path, filters={"config.group": group})
    # Get data
    data = [run.history()[metric] for run in runs]
    min_length = min([d.shape[0] for d in data])
    data = np.concatenate([datum.to_numpy()[:min_length,None] for datum in data], axis=-1)
    # Just get x-axis of one run since all runs should be identical
    x_axis = runs[0].history()[x_axis].to_numpy()[:min_length]

    # Filter out nans
    c_data, c_x_axis = [], []
    for datum, x in zip(data, x_axis):
        if np.sum(np.isnan(datum)) == 0:
            c_data += [datum]
            c_x_axis += [x]
    data, x_axis = np.array(c_data), np.array(c_x_axis)
    if prepend is not None:
        data, x_axis = prepend_missing_points(data, x_axis, prepend)
    return data, x_axis

def prepend_missing_points(data, x_axis, points):
    x_axis = np.concatenate([[0], x_axis])
    points = points[:data.shape[1]]
    points = np.reshape(points, [1,data.shape[1]])
    data = np.concatenate([points, data], axis=0)
    return data, x_axis

def plot(data, x_axis=None, min_x_axis=None, smooth=0., legend=None, color=None, marker=None):
    if x_axis is not None and min_x_axis is not None:
        # Take evenly spaced points to match mix_x_axis
        r = int(x_axis.shape[0]/min_x_axis.shape[0])
        indices = np.arange(0, x_axis.shape[0], r)
        #indices = list(filter(lambda idx: x_axis[idx] >= min_x_axis[0], indices))
        x_axis = x_axis[indices]
        data = data[indices]

    tsplot(data, x=x_axis, smooth=smooth, marker=marker, label=legend, color=color)

def plot_graph(project, groups, metrics, x_axes, save_name, xlim=None, ylim=None, legends=None, smooth=0.,
               colors=None, markers=None, horizontal_lines=None, horizontal_lines_colors=None, horizontal_lines_legends=None,
               horizontal_lines_markers=None, ylabel_length=None, prepend=None, x_label=None, y_label=None, correct_x_axis=False,
               show_legend=False):
    # Retrieve data
    metrics = [metrics]*len(groups) if type(metrics) != list else metrics
    x_axes = [x_axes]*len(groups) if type(x_axes) != list else x_axes
    data = [retrieve_group(project, *args, prepend=prepend) for args in zip(groups, metrics, x_axes)]

    # Take value at equally spaced intervals
    min_x_axis = min([x_axis for _, x_axis in data], key=lambda x: x.shape[0])

    # Plot any horizontal lines
    if horizontal_lines is not None:
        hcolors = [horizontal_lines_colors]*len(groups) if type(horizontal_lines_colors) != list else horizontal_lines_colors
        hmarkers = [horizontal_lines_markers]*len(groups) if type(horizontal_lines_markers) != list else horizontal_lines_markers
        for line, color, legend, marker in zip(horizontal_lines, hcolors, horizontal_lines_legends, hmarkers):
            plt.plot(min_x_axis, line*np.ones(min_x_axis.shape), linewidth=LINEWIDTH, marker=marker, markersize=MARKERSIZE, color=color, label=legend)

    # Plot data
    legends = [legends]*len(groups) if type(legends) != list else legends
    colors = [colors]*len(groups) if type(colors) != list else colors
    markers = [markers]*len(groups) if type(markers) != list else markers
    for (datum, x_axis), legend, color, marker in zip(data, legends, colors, markers):
        plot(datum, x_axis, min_x_axis, smooth, legend, color, marker)

    # Format plot
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if plt.yticks()[0][-1] >= 2000:
        ylabels = ['%d' % y + 'k' for y in plt.yticks()[0]/1000]
        plt.gca().set_yticklabels(ylabels)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    # Correct ylabels length (will cast to int)
    if ylabel_length is not None:
        ax = plt.gca()
        ylabels = [' '*(ylabel_length-len(str(int(y))))+'%d'%y for y in plt.yticks()[0]]
        ax.set_yticklabels(ylabels)
    plt.margins(x=0)
    plt.gca().grid(which='major', linestyle='-', linewidth='0.2', color='#d3d3d3')
    plt.grid('on')

    # Label axes
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if show_legend:
        plt.legend(loc='upper left', prop={'size': 12})

    # Save
    #plt.show()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=500)

    plt.close()


def plot_run(run, metric, x_axis, save_name, xlim=None, ylim=None, smooth=0.,
             color=None, marker=None, x_label=None, y_label=None):

    run = api.run(run)
    data = run.history()
    x, y = data[x_axis].to_numpy(), data[metric].to_numpy()
    if np.isnan(y[0]):
        y[0] = y[1]
    assert np.sum(np.isnan(y)) == 0
    tsplot(y[...,None], x=x, smooth=smooth, marker=marker, label=None, color=color)

    # Format plot
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.margins(x=0)

    # Label axes
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    # Save
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()



# ============================================================================
# What to plot?
# ============================================================================

def main_results(save_dir):
    project = 'spiderbot/Pruning/'
    smooth = 0.
    color = 'b'
    marker = None

    def vgg11():
        # VGG11 - coarse grained - b = 20
        sd = os.path.join(save_dir, 'vgg11_cg_20')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'pqmaopai', 'rollout/ep_rew_mean', 'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'pqmaopai', 'infos/action',        'time/total_timesteps', sd+'/action.png', None, [0,1],   0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'pqmaopai', 'train/average_cost',  'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'pqmaopai', 'train/nu',            'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG11 - fine grained - b = 20
        sd = os.path.join(save_dir, 'vgg11_fg_20')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'2mhj5icm', 'rollout/ep_rew_mean',        'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'2mhj5icm', 'infos/raw_effective_action', 'time/total_timesteps', sd+'/action.png', None, None,    0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'2mhj5icm', 'train/average_cost',         'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'2mhj5icm', 'train/nu',                   'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG11 - coarse grained - b = 10
        sd = os.path.join(save_dir, 'vgg11_cg_10')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'5jnzgeb9', 'rollout/ep_rew_mean', 'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'5jnzgeb9', 'infos/action',        'time/total_timesteps', sd+'/action.png', None, [0,1],   0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'5jnzgeb9', 'train/average_cost',  'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'5jnzgeb9', 'train/nu',            'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG11 - fine grained - b = 10
        sd = os.path.join(save_dir, 'vgg11_fg_10')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'3vyjnvym', 'rollout/ep_rew_mean',        'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'3vyjnvym', 'infos/raw_effective_action', 'time/total_timesteps', sd+'/action.png', None, None,    0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'3vyjnvym', 'train/average_cost',         'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'3vyjnvym', 'train/nu',                   'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')


    def vgg16():
        # VGG16 - coarse grained - b = 20
        sd = os.path.join(save_dir, 'vgg16_cg_20')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'mujmsm84', 'rollout/ep_rew_mean', 'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'mujmsm84', 'infos/action',        'time/total_timesteps', sd+'/action.png', None, [0,1],   0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'mujmsm84', 'train/average_cost',  'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'mujmsm84', 'train/nu',            'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG16 - fine grained - b = 20
        sd = os.path.join(save_dir, 'vgg16_fg_20')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'fz2ysv15', 'rollout/ep_rew_mean',        'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'fz2ysv15', 'infos/raw_effective_action', 'time/total_timesteps', sd+'/action.png', None, None,    0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'fz2ysv15', 'train/average_cost',         'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'fz2ysv15', 'train/nu',                   'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG16 - coarse grained - b = 10
        sd = os.path.join(save_dir, 'vgg16_cg_10')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'1g2axami', 'rollout/ep_rew_mean', 'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'1g2axami', 'infos/action',        'time/total_timesteps', sd+'/action.png', None, [0,1],   0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'1g2axami', 'train/average_cost',  'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'1g2axami', 'train/nu',            'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG16 - fine grained - b = 10
        sd = os.path.join(save_dir, 'vgg16_fg_10')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'33u58bx6', 'rollout/ep_rew_mean',        'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'33u58bx6', 'infos/raw_effective_action', 'time/total_timesteps', sd+'/action.png', None, None,    0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'33u58bx6', 'train/average_cost',         'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'33u58bx6', 'train/nu',                   'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

    def vgg19():
        # VGG19 - coarse grained - b = 20
        sd = os.path.join(save_dir, 'vgg19_cg_20')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'38y44hd6', 'rollout/ep_rew_mean', 'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'38y44hd6', 'infos/action',        'time/total_timesteps', sd+'/action.png', None, [0,1],   0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'38y44hd6', 'train/average_cost',  'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'38y44hd6', 'train/nu',            'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG19 - fine grained - b = 20
        sd = os.path.join(save_dir, 'vgg19_fg_20')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'32wbawov', 'rollout/ep_rew_mean',        'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'32wbawov', 'infos/raw_effective_action', 'time/total_timesteps', sd+'/action.png', None, None,    0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'32wbawov', 'train/average_cost',         'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'32wbawov', 'train/nu',                   'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG19 - coarse grained - b = 10
        sd = os.path.join(save_dir, 'vgg19_cg_10')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'2g0nvrw4', 'rollout/ep_rew_mean', 'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'2g0nvrw4', 'infos/action',        'time/total_timesteps', sd+'/action.png', None, [0,1],   0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'2g0nvrw4', 'train/average_cost',  'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'2g0nvrw4', 'train/nu',            'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')

        # VGG19 - fine grained - b = 10
        sd = os.path.join(save_dir, 'vgg19_fg_10')
        os.makedirs(sd, exist_ok=True)
        plot_run(project+'ns1q2oua', 'rollout/ep_rew_mean',        'time/total_timesteps', sd+'/reward.png', None, [0,100], 0., color, marker, 'Timesteps', 'Reward')
        plot_run(project+'ns1q2oua', 'infos/raw_effective_action', 'time/total_timesteps', sd+'/action.png', None, None,    0., color, marker, 'Timesteps', 'Mean action')
        plot_run(project+'ns1q2oua', 'train/average_cost',         'time/total_timesteps', sd+'/cost.png',   None, [0,100], 0., color, marker, 'Timesteps', 'Cost')
        plot_run(project+'ns1q2oua', 'train/nu',                   'time/total_timesteps', sd+'/nu.png',     None, None,    0., color, marker, 'Timesteps', 'Lagrange multiplier')




    #vgg11()
    #vgg16()
    vgg19()






if __name__=='__main__':
    start = time.time()
    main_results('pruning/plots')
    print('Time taken: ', (time.time()-start))

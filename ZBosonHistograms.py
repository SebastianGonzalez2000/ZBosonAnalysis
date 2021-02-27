mll = {
    # change plotting parameters
    'bin_width':1.5333,
    'num_bins':30,
    'xrange_min': 68,
    'xlabel':r'$\mathrm{m_{ll}}$ [GeV]',
    'log_y': False,

    # change aesthetic parameters if you want
    'y_label_x_position':-0.09, # 0.09 to the left of y axis
    'legend_loc': 'upper right',
    'log_top_margin':10000, # to decrease the separation between data and the top of the figure, remove a 0
    'linear_top_margin':1.1 # to decrease the separation between data and the top of the figure, pick a number closer to 1
}

hist_dict = {'mll': mll}



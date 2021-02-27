import uproot
import uproot_methods
import pandas as pd
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PolynomialModel, GaussianModel, DoniachModel, ExponentialGaussianModel, VoigtModel
import matplotlib.patches as mpatches  # for "Total SM & uncertainty" merged legend handle
from matplotlib.lines import Line2D  # for dashed line in legend
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, LogLocator, LogFormatterSciNotation  # for minor ticks
import scipy.stats
import os

import ZBosonSamples
import ZBosonCuts
import ZBosonHistograms
import infofile



class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1, 10]:
            return LogFormatterSciNotation.__call__(self, x, pos=None)
        else:
            return "{x:g}".format(x=x)


save_results = None # 'h5' or 'csv' or 'pickle' or None

load_histograms = False
store_histograms = False

lumi = 10  # 10 fb-1 for data_A,B,C,D

fraction = 0.0015 # reduce this is you want the code to run quicker

lumi_used = lumi * fraction

tuple_path = "/Users/sebastiangonzalez/Desktop/ATLAS_REX_Project/"  # Seb's address

stack_order = ['single top', 'W+jets', 'ttbar', 'Diboson']  # put smallest contribution first, then increase



def expand_columns(df): ## Ready
    for object_column in df.select_dtypes('object').columns:

        # expand df.object_column into its own dataframe
        object_column_df = df[object_column].apply(pd.Series)

        # rename each variable
        object_column_df = object_column_df.rename(columns=lambda x: object_column + '_' + str(x))

        # join the object_column dataframe back to the original dataframe
        df = pd.concat([df[:], object_column_df[:]], axis=1)
        df = df.drop(object_column, axis=1)

    return df

def read_sample(s): ## Ready
    print('Processing '+s+' samples')
    frames = []
    for val in ZBosonSamples.samples[s]['list']:

        # use this if the data has already been processed and you just want to plot from the saved csv file
        read_from_csv = False
        if read_from_csv:
            temp = pd.read_csv('resultsZBoson/dataframe_id_'+val+'.csv')
            frames.append(temp)
            continue

        read_from_pickle = False
        if read_from_pickle:
            temp = pd.read_pickle('resultsZBoson/dataframe_id_'+val+'.pkl')
            frames.append(temp)
            continue

        prefix = "MC/mc_"
        if s == 'data':
            prefix = "Data/"
        else: prefix += str(infofile.infos[val]["DSID"])+"."
        fileString = tuple_path+prefix+val+".2lep.root" # change ending depending on collection used, e.g. .4lep.root
        if fileString != "":
            temp = read_file(fileString,val)
            if not os.path.exists('resultsZBoson') and save_results!=None: os.makedirs('resultsZBoson')
            if save_results=='csv': temp.to_csv('resultsZBoson/dataframe_id_'+val+'.csv')
            if save_results=='pickle': temp.to_pickle('resultsZBoson/dataframe_id_'+val+'.pkl', protocol=2)
            if save_results=='h5' and len(temp.index)>0:
                temp = expand_columns(temp)
                temp.to_hdf('resultsZBoson/dataframe_id_'+val+'.h5',key='df',mode='w')
            frames.append(temp)
        else:
            print("Error: "+val+" not found!")
    data_s = pd.concat(frames)
    return data_s


def get_data_from_files(): ##Ready
    data = {}
    for s in ZBosonSamples.samples:
        data[s] = read_sample(s)

    return data

def calc_mll(lep_pt, lep_eta, lep_phi, lep_E):

    vector_1 = uproot_methods.TLorentzVector.from_ptetaphie(lep_pt[0], lep_eta[0], lep_phi[0], lep_E[0])
    vector_2 = uproot_methods.TLorentzVector.from_ptetaphie(lep_pt[1], lep_eta[1], lep_phi[1], lep_E[1])

    vector_12 = vector_1 + vector_2

    return vector_12.mag / 1000

def calc_weight(mcWeight,scaleFactor_PILEUP,scaleFactor_ELE,
                scaleFactor_MUON, scaleFactor_LepTRIGGER):
    return mcWeight*scaleFactor_PILEUP*scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER

def get_xsec_weight(totalWeight,sample):
    info = infofile.infos[sample]
    weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    weight *= totalWeight
    return weight

def mychisqr(resids,heights):
    errors = np.sqrt(heights).astype(int)
    errors[errors == 0] = 1 #should actually use Poisson errors below N~10
    return np.sum(np.square(resids/errors))


def read_file(path, sample):
    start = time.time()
    print("\tProcessing: " + sample)
    data_all = pd.DataFrame()
    mc = uproot.open(path)["mini"]
    numevents = uproot.numentries(path, "mini")

    for data in mc.iterate(["scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER", "scaleFactor_PILEUP", "mcWeight", "trigE", "trigM", "lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_z0", "lep_type",
                            "lep_isTightID", "lep_ptcone30", "lep_etcone20", "lep_charge",
                            "lep_trackd0pvunbiased", "lep_tracksigd0pvunbiased", "jet_n"], flatten=False, entrysteps=2500000, outputtype=pd.DataFrame,
                           entrystop=numevents * fraction):

        try:
            nIn = len(data.index)

            if 'data' not in sample:
                data['totalWeight'] = np.vectorize(calc_weight)(data.mcWeight,data.scaleFactor_PILEUP,data.scaleFactor_ELE,data.scaleFactor_MUON,data.scaleFactor_LepTRIGGER)
                data['totalWeight'] = np.vectorize(get_xsec_weight)(data.totalWeight,sample)

            data.drop(["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"],
                    axis=1, inplace=True)

            # Cut on number of leptons
            fail = data[np.vectorize(ZBosonCuts.cut_lep_n)(data.lep_n)].index
            data.drop(fail, inplace=True)

            # Preselection cut for electron/muon trigger
            fail = data[np.vectorize(ZBosonCuts.lepton_trigger)(data.trigE, data.trigM)].index
            data.drop(fail, inplace=True)

            # Both leptons are tight
            fail = data[np.vectorize(ZBosonCuts.lepton_is_tight)(data.lep_isTightID)].index
            data.drop(fail, inplace=True)

            # Both leptons are isolated and hard pT
            fail = data[np.vectorize(ZBosonCuts.lepton_isolated_hard_pt)(data.lep_pt, data.lep_ptcone30, data.lep_etcone20)].index
            data.drop(fail, inplace=True)

            # electron and muon selection
            fail = data[np.vectorize(ZBosonCuts.lepton_selection)(data.lep_type, data.lep_pt,data.lep_eta, data.lep_phi, data.lep_E, data.lep_trackd0pvunbiased, data.lep_tracksigd0pvunbiased, data.lep_z0)].index
            data.drop(fail, inplace=True)

            # Cut on oppositely charged leptons
            fail = data[np.vectorize(ZBosonCuts.cut_opposite_charge)(data.lep_charge)].index
            data.drop(fail, inplace=True)

            # Cut on leptons of same flavour
            fail = data[np.vectorize(ZBosonCuts.cut_same_flavour)(data.lep_type)].index
            data.drop(fail, inplace=True)

            # Calculate invariant mass
            data['mll'] = np.vectorize(calc_mll)(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_E)

            # Cut on invariant mass
            fail = data[np.vectorize(ZBosonCuts.cut_invariant_mass)(data.mll)].index
            data.drop(fail, inplace=True)

            # jet cut
            fail = data[np.vectorize(ZBosonCuts.cut_jet_n)(data.jet_n)].index
            data.drop(fail, inplace=True)

            nOut = len(data.index)
            data_all = data_all.append(data)
            elapsed = time.time() - start
            print("\t\tTime taken: " + str(elapsed) + ", nIn: " + str(nIn) + ", nOut: " + str(nOut))
        except ValueError:
            print("ValueError. Probably vectorizing on zero-length input")
            continue

    return data_all


def plot_data(data):
    signal_format = 'hist'  # 'line' or 'hist' or None
    Total_SM_label = False  # for Total SM black line in plot and legend
    plot_label = r'$Z \rightarrow ll$'
    signal_label = plot_label


    signal = None
    for s in ZBosonSamples.samples.keys():
        if s not in stack_order and s != 'data': signal = s

    for x_variable, hist in ZBosonHistograms.hist_dict.items():

        h_bin_width = hist['bin_width']
        h_num_bins = hist['num_bins']
        h_xrange_min = hist['xrange_min']
        h_xlabel = hist['xlabel']
        h_log_y = hist['log_y']
        h_y_label_x_position = hist['y_label_x_position']
        h_legend_loc = hist['legend_loc']
        h_log_top_margin = hist[
            'log_top_margin']  # to decrease the separation between data and the top of the figure, remove a 0
        h_linear_top_margin = hist[
            'linear_top_margin']  # to decrease the separation between data and the top of the figure, pick a number closer to 1

        bins = [h_xrange_min + x * h_bin_width for x in range(h_num_bins + 1)]
        bin_centres = [h_xrange_min + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

        if store_histograms:
            stored_histos = {}

        if load_histograms: # not doing line for now
            npzfile = np.load(f'histograms/{x_variable}_hist_{fraction}.npz')
            # load bins
            loaded_bins = npzfile['bins']
            if not np.array_equal(bins, loaded_bins):
                print('Bins mismatch. That\'s a problem')
                raise Exception

            # load data
            data_x = npzfile['data']
            data_x_errors = np.sqrt(data_x)
            # load weighted signal
            signal_x_reshaped = npzfile[signal]
            signal_color = ZBosonSamples.samples[signal]['color']
            # load backgrounds
            mc_x_heights_list = []
            # mc_weights = []
            mc_colors = []
            mc_labels = []
            mc_x_tot = np.zeros(len(bin_centres))
            for s in stack_order:
                if not s in npzfile: continue
                mc_labels.append(s)
                # mc_x.append(data[s][x_variable].values)
                mc_colors.append(ZBosonSamples.samples[s]['color'])
                # mc_weights.append(data[s].totalWeight.values)
                mc_x_heights = npzfile[s]
                mc_x_heights_list.append(mc_x_heights)
                mc_x_tot = np.add(mc_x_tot, mc_x_heights)
            mc_x_err = np.sqrt(mc_x_tot)

        else:
            # ======== This creates histograms for the raw data events ======== #
            # no weights necessary (it's data)
            data_x, _ = np.histogram(data['data'][x_variable].values, bins=bins)
            data_x_errors = np.sqrt(data_x)
            if store_histograms: stored_histos['data'] = data_x # saving histograms for later loading

            # ======== This creates histograms for signal simulation (Z->ll) ======== #
            # need to consider the event weights here
            signal_x = None
            if signal_format == 'line':
                signal_x, _ = np.histogram(data[signal][x_variable].values, bins=bins,
                                        weights=data[signal].totalWeight.values)
            elif signal_format == 'hist':
                signal_x = data[signal][x_variable].values
                signal_weights = data[signal].totalWeight.values
                signal_color = ZBosonSamples.samples[signal]['color']
                signal_x_reshaped, _ = np.histogram(data[signal][x_variable].values, bins=bins,
                                        weights=data[signal].totalWeight.values)
                if store_histograms: stored_histos[signal] = signal_x_reshaped # saving histograms for later loading

            # ======== This creates histograms for all of the background simulation ======== #
            # weights are also necessary here, since we produce an arbitrary number of MC events
            mc_x_heights_list = []
            mc_weights = []
            mc_colors = []
            mc_labels = []
            mc_x_tot = np.zeros(len(bin_centres))

            for s in stack_order:
                if not s in data: continue
                if data[s].empty: continue
                mc_labels.append(s)
                # mc_x.append(data[s][x_variable].values)
                mc_colors.append(ZBosonSamples.samples[s]['color'])
                mc_weights.append(data[s].totalWeight.values)
                mc_x_heights, _ = np.histogram(data[s][x_variable].values, bins=bins, weights=data[s].totalWeight.values) #mc_heights?
                mc_x_heights_list.append(mc_x_heights)
                mc_x_tot = np.add(mc_x_tot, mc_x_heights)
                if store_histograms: stored_histos[s] = mc_x_heights #saving histograms for later loading

            mc_x_err = np.sqrt(mc_x_tot)

        data_x_without_bkg = data_x - mc_x_tot

        # data fit

        # get rid of zero errors (maybe messy) : TODO a better way to do this?
        for i, e in enumerate(data_x_errors):
            if e == 0: data_x_errors[i] = np.inf
        if 0 in data_x_errors:
            print('please don\'t divide by zero')
            raise Exception

        bin_centres_array = np.asarray(bin_centres)

        # *************
        # Models
        # *************

        doniach_mod = DoniachModel()
        pars_doniach = doniach_mod.guess(data_x_without_bkg, x=bin_centres_array, amplitude=2100000 * fraction, center=90.5, sigma=2.3, height=10000 * fraction / 0.01, gamma=0)
        doniach = doniach_mod.fit(data_x_without_bkg, pars_doniach, x=bin_centres_array, weights=1 / data_x_errors)
        params_dict_doniach = doniach.params.valuesdict()


        gaussian_mod = GaussianModel()
        pars_gaussian = gaussian_mod.guess(data_x_without_bkg, x=bin_centres_array, amplitude=6000000*fraction, center=90.5, sigma=3)
        gaussian = gaussian_mod.fit(data_x_without_bkg, pars_gaussian, x=bin_centres_array, weights=1 / data_x_errors)
        params_dict_gaussian = gaussian.params.valuesdict()


        exponential_gaussian_mod = ExponentialGaussianModel()
        pars = exponential_gaussian_mod.guess(data_x_without_bkg, x=bin_centres_array, amplitude=6000000*fraction, center = 90.5, sigma=2.9, gamma = 1)
        exp_gaussian = exponential_gaussian_mod.fit(data_x_without_bkg, pars, x=bin_centres_array, weights=1 / data_x_errors)
        params_dict_exp_gaussian = exp_gaussian.params.valuesdict()


        voigt_mod = VoigtModel()
        pars = voigt_mod.guess(data_x_without_bkg, x=bin_centres_array, amplitude=6800000*fraction, center=90.5, sigma=1.7)
        voigt = voigt_mod.fit(data_x_without_bkg, pars, x=bin_centres_array, weights=1 / data_x_errors)
        params_dict_voigt = voigt.params.valuesdict()

        if store_histograms:
            # save all histograms in npz format. different file for each variable. bins are common
            os.makedirs('histograms', exist_ok=True)
            np.savez(f'histograms/{x_variable}_hist.npz', bins=bins, **stored_histos)
            # ======== Now we start doing the fit ======== #



        # *************
        # Main plot
        # *************
        plt.clf()
        plt.axes([0.1, 0.3, 0.85, 0.65])  # (left, bottom, width, height)
        main_axes = plt.gca()
        main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors, fmt='ko', label='Data')
        # this effectively makes a stacked histogram
        bottoms = np.zeros_like(bin_centres)
        for mc_x_height, mc_color, mc_label in zip(mc_x_heights_list, mc_colors, mc_labels) :
            main_axes.bar(bin_centres, mc_x_height, bottom=bottoms, color=mc_color, label=mc_label, width=h_bin_width*1.01)
            bottoms = np.add(bottoms, mc_x_height)


        main_axes.plot(bin_centres, doniach.best_fit, '-r')
        main_axes.plot(bin_centres, gaussian.best_fit, '-g')
        main_axes.plot(bin_centres, exp_gaussian.best_fit, '-y')
        main_axes.plot(bin_centres, voigt.best_fit, '-p')

        #mc_heights = main_axes.hist(mc_x, bins=bins, weights=mc_weights, stacked=True, color=mc_colors, label=mc_labels)
        if Total_SM_label:
            totalSM_handle, = main_axes.step(bins, np.insert(mc_x_tot, 0, mc_x_tot[0]), color='black')
        if signal_format == 'line':
            main_axes.step(bins, np.insert(signal_x, 0, signal_x[0]), color=ZBosonSamples.samples[signal]['color'],
                           linestyle='--',
                           label=signal)
        elif signal_format == 'hist':
            main_axes.bar(bin_centres, signal_x_reshaped, bottom=bottoms, color=signal_color, label=signal,
                          width=h_bin_width*1.01)
            bottoms = np.add(bottoms, signal_x_reshaped)
        main_axes.bar(bin_centres, 2 * mc_x_err, bottom=bottoms - mc_x_err, alpha=0.5, color='none', hatch="////",
                      width=h_bin_width*1.01, label='Stat. Unc.')

        mc_x_tot = bottoms

        main_axes.set_xlim(left=h_xrange_min, right=bins[-1])
        main_axes.xaxis.set_minor_locator(AutoMinorLocator())  # separation of x axis minor ticks
        main_axes.tick_params(which='both', direction='in', top=True, labeltop=False, labelbottom=False, right=True,
                              labelright=False)

        if h_log_y:
            main_axes.set_yscale('log')
            smallest_contribution = mc_x_heights_list[0] # TODO: mc_heights or mc_x_heights
            smallest_contribution.sort()
            bottom = smallest_contribution[-2]
            if bottom == 0: bottom = 0.001 # log doesn't like zero
            top = np.amax(data_x) * h_log_top_margin
            main_axes.set_ylim(bottom=bottom, top=top)
            main_axes.yaxis.set_major_formatter(CustomTicker())
            locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
            main_axes.yaxis.set_minor_locator(locmin)
        else:
            main_axes.set_ylim(bottom=0, top=(np.amax(data_x) + math.sqrt(np.amax(data_x))) * h_linear_top_margin)
            main_axes.yaxis.set_minor_locator(AutoMinorLocator())
            main_axes.yaxis.get_major_ticks()[0].set_visible(False)

        plt.text(0.015, 0.97, 'ATLAS Open Data', ha="left", va="top", family='sans-serif', transform=main_axes.transAxes,
                 fontsize=13)
        plt.text(0.015, 0.9, 'for education', ha="left", va="top", family='sans-serif', transform=main_axes.transAxes,
                 style='italic', fontsize=8)
        plt.text(0.015, 0.86, r'$\sqrt{s}=13\,\mathrm{TeV},\;\int L\,dt=$' + str(lumi_used) + '$\,\mathrm{fb}^{-1}$',
                 ha="left", va="top", family='sans-serif', transform=main_axes.transAxes)
        plt.text(0.015, 0.78, plot_label, ha="left", va="top", family='sans-serif', transform=main_axes.transAxes)
        plt.text(0.015, 0.72, r'$m_Z = $' + str(round(params_dict_doniach['center'], 4)) + ' GeV', ha="left", va="top", family='sans-serif', transform=main_axes.transAxes,
                 fontsize=10)

        # Create new legend handles but use the colors from the existing ones
        handles, labels = main_axes.get_legend_handles_labels()
        if signal_format == 'line':
            handles[labels.index(signal)] = Line2D([], [], c=ZBosonSamples.samples[signal]['color'], linestyle='dashed')
        uncertainty_handle = mpatches.Patch(facecolor='none', hatch='////')
        if Total_SM_label:
            handles.append((totalSM_handle, uncertainty_handle))
            labels.append('Total SM')
        else:
            handles.append(uncertainty_handle)
            labels.append('Stat. Unc.')

        # specify order within legend
        new_handles = [handles[labels.index('Data')]]
        new_labels = ['Data']
        for s in reversed(stack_order):
            if s not in labels:
                continue
            new_handles.append(handles[labels.index(s)])
            new_labels.append(s)
        if signal is not None:
            new_handles.append(handles[labels.index(signal)])
            new_labels.append(signal_label)
        if Total_SM_label:
            new_handles.append(handles[labels.index('Total SM')])
            new_labels.append('Total SM')
        else:
            new_handles.append(handles[labels.index('Stat. Unc.')])
            new_labels.append('Stat. Unc.')
        main_axes.legend(handles=new_handles, labels=new_labels, frameon=False, loc=h_legend_loc)


        # *************
        # Data / MC plot
        # *************


        plt.axes([0.1, 0.1, 0.85, 0.2])  # (left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.yaxis.set_major_locator(MaxNLocator(nbins='auto', symmetric=True))
        ratio_axes.errorbar(x=bin_centres, y=data_x / signal_x_reshaped, fmt='ko') # TODO: yerr=data_x_errors produce error bars that are too big
        ratio_axes.set_xlim(left=h_xrange_min, right=bins[-1])
        ratio_axes.plot(bins, np.ones(len(bins)), color='k')
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())  # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9, -0.2)  # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(h_xlabel, fontname='sans-serif', fontsize=11)
        ratio_axes.set_ylim(bottom=0, top=2)
        ratio_axes.set_yticks([0, 1])
        ratio_axes.tick_params(which='both', direction='in', top=True, labeltop=False, right=True, labelright=False)
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        ratio_axes.set_ylabel(r'Data / Pred', fontname='sans-serif', x=1, fontsize=11)


        # Generic features for both plots
        main_axes.yaxis.set_label_coords(h_y_label_x_position, 1)
        ratio_axes.yaxis.set_label_coords(h_y_label_x_position, 0.5)

        plt.savefig("ZBoson_" + x_variable + ".pdf", bbox_inches='tight')

        # ========== Statistics ==========

        # ========== Doniach ==========
        chisqr_doniach = mychisqr(doniach.residual, data_x)
        redchisqr_doniach = chisqr_doniach/doniach.nfree
        center_doniach = params_dict_doniach['center']
        sigma_doniach = params_dict_doniach['sigma']
        amplitude_doniach = params_dict_doniach['amplitude']

        # ========== Gaussian ==========
        chisqr_gaussian = mychisqr(gaussian.residual, data_x)
        redchisqr_gaussian = chisqr_gaussian / gaussian.nfree
        center_gaussian = params_dict_gaussian['center']
        sigma_gaussian = params_dict_gaussian['sigma']
        amplitude_gaussian = params_dict_gaussian['amplitude']

        # ========== Exponential Gaussian ==========
        chisqr_exp_gaussian = mychisqr(exp_gaussian.residual, data_x)
        redchisqr_exp_gaussian = chisqr_exp_gaussian / exp_gaussian.nfree
        center_exp_gaussian = params_dict_exp_gaussian['center']
        sigma_exp_gaussian = params_dict_exp_gaussian['sigma']
        amplitude_exp_gaussian = params_dict_exp_gaussian['amplitude']

        # ========== Voigt ==========
        chisqr_voigt = mychisqr(voigt.residual, data_x)
        redchisqr_voigt = chisqr_voigt / voigt.nfree
        center_voigt = params_dict_voigt['center']
        sigma_voigt = params_dict_voigt['sigma']
        amplitude_voigt = params_dict_voigt['amplitude']


        df_dict = {'fraction':[fraction],
                   'luminosity':[lumi_used],
                   'doniach chisqr':[chisqr_doniach],
                       'doniach redchisqr':[redchisqr_doniach],
                       'gaussian chisqr':[chisqr_gaussian],
                       'gaussian redchisqr':[redchisqr_gaussian],
                       'exponential gaussian chisqr':[chisqr_exp_gaussian],
                       'exponential gaussian redchisqr':[redchisqr_exp_gaussian],
                       'voigt chisqr':[chisqr_voigt],
                       'voigt redchisqr':[redchisqr_voigt]}

        temp = pd.DataFrame(df_dict)

        fit_results = pd.read_csv('fit_results.csv')

        fit_results_concat = pd.concat([fit_results, temp])

        fit_results_concat.to_csv('fit_results.csv', index = False)


        print("=====================================================")
        print("Statistics for the Doniach Model: ")
        print("\n")
        print("chi^2 = " + str(chisqr_doniach))
        print("chi^2/dof = " + str(redchisqr_doniach))
        print("center = " + str(center_doniach))
        print("sigma = " + str(sigma_doniach))
        print("amplitude = " + str(amplitude_doniach))
        print("height = " + str(params_dict_doniach['height']))

        print("\n")
        print("=====================================================")
        print("Statistics for the Gaussian Model: ")
        print("\n")
        print("chi^2 = " + str(chisqr_gaussian))
        print("chi^2/dof = " + str(redchisqr_gaussian))
        print("center = " + str(center_gaussian))
        print("sigma = " + str(sigma_gaussian))
        print("amplitude = " + str(amplitude_gaussian))

        print("\n")
        print("=====================================================")
        print("Statistics for the Exponential Gaussian Model: ")
        print("\n")
        print("chi^2 = " + str(chisqr_exp_gaussian))
        print("chi^2/dof = " + str(redchisqr_exp_gaussian))
        print("center = " + str(center_exp_gaussian))
        print("sigma = " + str(sigma_exp_gaussian))
        print("amplitude = " + str(amplitude_exp_gaussian))

        print("\n")
        print("=====================================================")
        print("Statistics for the Voigt Model: ")
        print("\n")
        print("chi^2 = " + str(chisqr_voigt))
        print("chi^2/dof = " + str(redchisqr_voigt))
        print("center = " + str(center_voigt))
        print("sigma = " + str(sigma_voigt))
        print("amplitude = " + str(amplitude_voigt))

    if load_histograms: return None, None
    return signal_x, mc_x_tot


if __name__ == "__main__":
    start = time.time()
    if not load_histograms:
        data = get_data_from_files()
        signal_yields, background_yields = plot_data(data)
    else:
        plot_data(None)
    elapsed = time.time() - start
    print("Time taken: "+str(elapsed))


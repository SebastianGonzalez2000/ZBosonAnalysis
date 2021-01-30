import math
import uproot_methods


def cut_lep_n(lep_n):
    return lep_n != 2


def cut_opposite_charge(lep_charge):
    return lep_charge[0] * lep_charge[1] > 0


def cut_same_flavour(lep_type):
    return lep_type[0] != lep_type[1]


def lepton_trigger(trigE, trigM):
    return not (trigE or trigM)


def lepton_is_tight(lep_isTightID):
    return not (lep_isTightID[0] and lep_isTightID[1])


def lepton_isolated_hard_pt(lep_pt, lep_ptcone30, lep_etcone20):

    lep_0 = (lep_pt[0]>25000 and ((lep_ptcone30[0]/lep_pt[0])<0.15 and ((lep_etcone20[0]/lep_pt[0])<0.15)))
    lep_1 = (lep_pt[1] > 25000 and ((lep_ptcone30[1] / lep_pt[1]) < 0.15 and ((lep_etcone20[1] / lep_pt[1]) < 0.15)))

    return not (lep_0 and lep_1)


def cut_invariant_mass(mll):
    return math.fabs(mll - 91.18) >= 25.


def cut_jet_n(jet_n):
    return jet_n != 0


def lepton_selection(lep_type, lep_pt, lep_eta, lep_phi, lep_E, lep_trackd0pvunbiased, lep_tracksigd0pvunbiased, lep_z0):

    lepTemp_0 = uproot_methods.TLorentzVector.from_ptetaphie(lep_pt[0]/1000., lep_eta[0], lep_phi[0], lep_E[0]/1000.)
    lepTemp_1 = uproot_methods.TLorentzVector.from_ptetaphie(lep_pt[1] / 1000., lep_eta[1], lep_phi[1], lep_E[1] / 1000.)

    electron_selection_0 = (lep_type[0]==11 and math.fabs(lep_eta[0])<2.47 and (math.fabs(lep_eta[0])<1.37 or math.fabs(lep_eta[0])>1.52)) and not ((math.fabs(lep_trackd0pvunbiased[0])/lep_tracksigd0pvunbiased[0]) < 5 and math.fabs(lep_z0[0]*math.sin(lepTemp_0.theta()))<0.5)
    electron_selection_1 = (lep_type[1] == 11 and math.fabs(lep_eta[1]) < 2.47 and (math.fabs(lep_eta[1]) < 1.37 or math.fabs(lep_eta[1]) > 1.52)) and not ((math.fabs(lep_trackd0pvunbiased[1]) / lep_tracksigd0pvunbiased[1]) < 5 and math.fabs(lep_z0[1] * math.sin(lepTemp_1.theta())) < 0.5)
    electron_selection = electron_selection_0 and electron_selection_1

    muon_selection_0 = (lep_type[0]==13 and math.fabs(lep_eta[0])<2.5) and not ((math.fabs(lep_trackd0pvunbiased[0])/lep_tracksigd0pvunbiased[0])<3 and math.fabs(lep_z0[0]*math.sin(lepTemp_0.theta))<0.5)
    muon_selection_1 = (lep_type[1] == 13 and math.fabs(lep_eta[1]) < 2.5) and not ((math.fabs(lep_trackd0pvunbiased[1]) / lep_tracksigd0pvunbiased[1]) < 3 and math.fabs(lep_z0[1] * math.sin(lepTemp_1.theta)) < 0.5)
    muon_selection = muon_selection_0 and muon_selection_1

    return electron_selection or muon_selection



import math

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

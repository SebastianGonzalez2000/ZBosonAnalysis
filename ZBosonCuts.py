# Cut on number of leptons
def cut_lep_n(lep_n):
    return lep_n != 2

# Cut on lepton reconstruction
def cut_lep_reconstruction(lep_isTightID):
    return lep_isTightID[0] == False or lep_isTightID[1] == False

# Cut on Transverse momentum
def cut_lep_pt(lep_pt):
    # first lepton is [0], 2nd lepton is [1] etc
    return lep_pt[0] < 25000 or lep_pt[1] < 25000

# Cut on lepton momentum isolation
def cut_lep_isolation_ptcone30(lep_ptcone30, lep_pt):
    return lep_ptcone30[0]/lep_pt[0] >= 0.15 or lep_ptcone30[1]/lep_pt[1] >= 0.15

# Cut on lepton energy isolation
def cut_lep_isolation_etcone20(lep_etcone20, lep_pt):
    return lep_etcone20[0]/lep_pt[0] >= 0.15 or lep_etcone20[0]/lep_pt[0] >= 0.15

# Cut on electron pseudorapidity outside the fiducial region
def cut_electron_eta_fiducial(lep_eta, lep_type):
    cut_lep_one = not (lep_type[0]==11 and abs(lep_eta[0])<2.47 and (abs(lep_eta[0]<1.37) or abs(lep_eta[0]) > 1.52))
    cut_lep_two = not (lep_type[1] == 11 and abs(lep_eta[1]) < 2.47 and (abs(lep_eta[1] < 1.37) or abs(lep_eta[1]) > 1.52))
    return cut_lep_one or cut_lep_two

# Cut on muon pseudorapidity outside the fiducial region
def cut_muon_eta_fiducial(lep_eta, lep_type):
    cut_lep_one = not (lep_type[0] == 13 and abs(lep_eta[0]) < 2.5)
    cut_lep_two = not (lep_type[1] == 13 and abs(lep_eta[1]) < 2.5)
    return cut_lep_one or cut_lep_two

# Cut on leptons having opposite charges
def cut_opposite_charge(lep_charge):
    return lep_charge[0] * lep_charge[1] > 0

# Cut on leptons having the same flavour
def cut_same_flavour(lep_type):
    return lep_type[0] != lep_type[1]










# Cut on reconstructed invariant mass lower limit
# paper: "in the diphoton invariant mass range between 100 GeV and 160 GeV"
def cut_mass_lower(myy):
# want to discard minimum invariant reconstructed mass < 100 GeV
    return myy < 100

# Cut on reconstructed invariant mass upper limit
# paper: "in the diphoton invariant mass range between 100 GeV and 160 GeV"
def cut_mass_upper(myy):
# want to discard maximum invariant reconstructed mass > 160 GeV
    return myy > 160
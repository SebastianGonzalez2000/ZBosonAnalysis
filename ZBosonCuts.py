# Cut on number of leptons
def cut_lepton_n (lep_n):
    return lep_n != 2

# Cut on pseudorapidity outside the fiducial region
def cut_lep_eta_fiducial(lep_eta):
    return lep_eta[0] > 2.37 or lep_eta[1] > 2.37 or lep_eta[0] < -2.37 or lep_eta[1] < -2.37

# Cut on pseudorapidity in barrel/end-cap transition region
# paper: "excluding the calorimeter barrel/end-cap transition region 1.37 < |η| < 1.52"
def cut_lep_eta_transition(lep_eta):
    if lep_eta[0] < 1.52 and lep_eta[0] > 1.37: return True
    elif lep_eta[1] < 1.52 and lep_eta[1] > 1.37: return True
    elif lep_eta[0] > -1.52 and lep_eta[0] < -1.37: return True
    elif lep_eta[1] < -1.37 and lep_eta[1] > -1.52: return True
    else: return False

# Cut on Transverse momentum
# paper: "The leading (sub-leading) photon candidate is required to have ET > 40 GeV (30 GeV)"
def cut_lep_pt(lep_pt):
# want to discard any events where photon_pt[0] < 40000 MeV or photon_pt[1] < 30000 MeV
    # first lepton is [0], 2nd lepton is [1] etc
    return lep_pt[0] < 40000 or lep_pt[1] < 30000

# Cut on photon reconstruction
# paper: "Photon candidates are required to pass identification criteria"
def cut_photon_reconstruction(photon_isTightID):
# want to discard events where it is false for one or both photons
    return photon_isTightID[0] == False or photon_isTightID[1] == False

# Cut on energy isolation
# paper: "Photon candidates are required to have an isolation transverse energy of less than 4 GeV"
def cut_isolation_et(photon_etcone20):
# want to discard events where isolation eT > 4000 MeV
    return photon_etcone20[0] > 4000 or photon_etcone20[1] > 4000

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
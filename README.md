## Data Analysis of Higgs Interaction from different samples 

## Guide to Datasets
### Datasets of Higgs Decay using various alterative of SM & SMEFT
### 
## Higgs candidate (reconstructed)

### Lumi Weight - How many real events this single simulated event corresponds to, for a given luminosity

### Higgs_Mass - Invarient Mass of the reconstructed Higgs Candidate (Should peak around 125 GeV)
### Higgs_pT - Transverse momentum of the Higgs (Sensitive to: production mode, higher-dimension SMEFT operators)
### Higgs_Eta - Pseudorapidity of the Higgs (Tells you how forward / central it is)
### Higgs_Phi - Azimuthal angle in the transverse plane

## bb system (H → b b̄)
### DPhi_bb - angle between the two-jets, Sensitive to spin and event topology
### m_bbyy - Invariant mass of the bbγγ system
### nBTaggedJets - Number of jets tagged as b-jets

## γγ system (H → γγ)
### LeadPhoton_pT, SubLeadPhoton_pT - Transverse momenta of the two photons, Ordered by pT
### LeadPhoton_Eta, SubLeadPhoton_Eta - Photon directions
### LeadPhoton_Phi, SubLeadPhoton_Phi - Photon azimuthal angles

## Jets & VBF-like structure
### LeadJet_*, SubLeadJet_* - Highest-pT and second-highest-pT jets (Variables: pT, Eta, Phi, M)
### M_jj - Invariant mass of the two leading jets (Large values → VBF-like events)
### Eta_jj - angle between the two leading jets
### pT_jj - Transverse momentum of the dijet system
### Phi_jj - Azimuthal angle of the dijet system
### signed_DeltaPhi_jj - Signed Δφ between jets (Used in CP studies)

## Angular / spin-correlation observables 
### cosThetaStar - Scattering angle of the Higgs pair (Sensitive to production dynamics)
### costheta1, costheta2 - Decay angles in the Higgs rest frames (Sensitive to: CP properties anomalous couplings)

## Labels & weights
### is_HiggsEvent - Boolean Flag (True -> signal, False -> background)
### Lumi_weight - Event Weight, (Encodes: cross section, generator weight, luminosity scaling)
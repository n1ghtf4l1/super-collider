#### **super-collider**

# **Finding τ → μμμ decay in collisions of accelerated particles**

##### Predict τ → 3μ decay in acelerated particle collisions.

### **Overview**

The aim of this playground challenge is to find a phenomenon that is not already known to exist – charged lepton flavour violation – thereby helping to establish "[new physics](https://en.wikipedia.org/wiki/Physics_beyond_the_Standard_Model)". 

The laws of nature ensure that some physical quantities, such as energy or momentum, are conserved. From [Noether’s theorem](https://en.wikipedia.org/wiki/Noether%27s_theorem), we know that each conservation law is associated with a fundamental symmetry. For example, conservation of energy is due to the time-invariance (the outcome of an experiment would be the same today or tomorrow) of physical systems. The fact that physical systems behave the same, regardless of where they are located or how they are oriented, gives rise to the conservation of linear and angular momentum.

Symmetries are also crucial to the structure of the [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) of particle physics, present theory of interactions at microscopic scales. Some are built into the model, while others appear accidentally from it. In the Standard Model, lepton flavour, the number of electrons and electron-neutrinos, muons and muon-neutrinos, and tau and tau-neutrinos, is one such conserved quantity.

Interestingly, in many proposed extensions to the Standard Model, this symmetry doesn’t exist, implying decays that do not conserve lepton flavour are possible. One decay searched for at the LHC is τ → μμμ (or τ → 3μ). Observation of this decay would be a clear indication of the violation of lepton flavour and a sign of long-sought new physics.

With real data from the [LHCb experiment](http://lhcb-public.web.cern.ch/lhcb-public/) at the LHC, mixed with simulated datasets of the decay, the metric used includes checks that physicists do in their analysis to make sure the results are unbiased. These checks have been built to help ensure that the results will be useful for physicists in future studies.

Review the [Data Page](https://www.kaggle.com/c/flavours-of-physics-kernels-only/data) and download the [Starter Kit](https://github.com/yandexdataschool/flavours-of-physics-start).

### **Dataset Description**

**NOTE:** Due to large dataset size not possible to push from git, Complete of dataset available [here](https://huggingface.co/datasets/n1ghtf4l1/super-collider). 

In this repository, given a list of collision events and their properties. task is to predict whether a τ → 3μ decay happened in this collision. This τ → 3μ is currently assumed by scientists not to happen, and the goal of this [task](https://www.kaggle.com/competitions/flavours-of-physics-kernels-only/overview) is to discover τ → 3μ happening more frequently than scientists currently can understand.

It is challenging to design a machine learning problem for something never observed before. Scientists at CERN developed the following designs to achieve the goal.

#### **training.csv**

This is a labelled dataset (the label ‘signal’ being ‘1’ for signal events, ‘0’ for background events) to train the classifier. Signal events have been simulated, while background events are real data.

This real data is collected by the LHCb detectors observing collisions of accelerated particles with a specific mass range in which τ → 3μ can’t happen. We call these events “background” and label them 0.
- FlightDistance - Distance between τ and PV (primary vertex, the original protons collision point).
- FlightDistanceError - Error on FlightDistance.
- mass - reconstructed τ candidate invariant mass, which is **absent in the test samples.**
- LifeTime - Life time of tau candidate.
- IP - Impact Parameter of tau candidate.
- IPSig - Significance of Impact Parameter.
- VertexChi2 - χ2 of τ vertex.
- dira - Cosine of the angle between the τ momentum and line between PV and tau vertex. 
- pt - transverse momentum of τ.
- DOCAone - Distance of Closest Approach between p0 and p1.
- DOCAtwo - Distance of Closest Approach between p1 and p2.
- DOCAthree - Distance of Closest Approach between p0 and p2.
- IP_p0p2 - Impact parameter of the p0 and p2 pair.
- IP_p1p2 - Impact parameter of the p1 and p2 pair.
- isolationa - Track isolation variable.
- isolationb - Track isolation variable.
- isolationc - Track isolation variable.
- isolationd - Track isolation variable.
- isolatione - Track isolation variable.
- isolationf - Track isolation variable.
- iso - Track isolation variable.
- CDF1 - Cone isolation variable.
- CDF2 - Cone isolation variable.
- CDF3 - Cone isolation variable.
- production - source of τ. This variable is **absent in the test samples.**
- ISO_SumBDT - Track isolation variable.
- p0_IsoBDT - Track isolation variable.
- p1_IsoBDT - Track isolation variable.
- p2_IsoBDT - Track isolation variable.
- p0_track_Chi2Dof - Quality of p0 muon track.
- p1_track_Chi2Dof - Quality of p1 muon track.
- p2_track_Chi2Dof - Quality of p2 muon track.
- p0_pt - Transverse momentum of p0 muon.
- p0_p - Momentum of p0 muon.
- p0_eta - Pseudorapidity of p0 muon.
- p0_IP - Impact parameter of p0 muon.
- p0_IPSig - Impact Parameter Significance of p0 muon.
- p1_pt - Transverse momentum of p1 muon.
- p1_p - Momentum of p1 muon.
- p1_eta - Pseudorapidity of p1 muon.
- p1_IP - Impact parameter of p1 muon.
- p1_IPSig - Impact Parameter Significance of p1 muon.
- p2_pt - Transverse momentum of p2 muon.
- p2_p - Momentum of p2 muon.
- p2_eta - Pseudorapidity of p2 muon.
- p2_IP - Impact parameter of p2 muon.
- p2_IPSig - Impact Parameter Significance of p2 muon.
- SPDhits - Number of hits in the SPD detector.
- min_ANNmuon - Muon identification. LHCb collaboration trains Artificial Neural Networks (ANN) from informations from RICH, ECAL, HCAL, Muon system to distinguish muons from other particles. This variables denotes the minimum of the three muons ANN. min ANNmuon should not be used for training. **This variable is absent in the test samples.**
- **signal - This is the target variable for you to predict in the test samples.**

#### **test.csv**

The test dataset has all the columns that training.csv has, except **mass, production, min_ANNmuon,** and **signal.** 

The test dataset consists of a few parts:
- simulated signal events for the τ → 3μ
- real background data for the τ → 3μ
- simulated events for the control channel, (ignored for scoring, used by agreement test)
- real data for the control channel (ignored for scoring, used by agreement test)

It is required to submit predictions for ALL the test entries. We will need to treat them all the same and predict as if they are all the same channel's collision events. 

A submission is only scored after passing both the [agreement test](https://www.kaggle.com/c/flavours-of-physics/details/agreement-test) and the [correlation test](https://www.kaggle.com/c/flavours-of-physics/details/correlation-test).

#### **check_agreement.csv: Ds → φπ data**

This dataset contains simulated and real events from the Control channel Ds → φπ to evaluate your simulated-real data of submission agreement locally. It contains the same columns as test.csv and weight column. For more details see [agreement test](https://www.kaggle.com/c/flavours-of-physics/details/agreement-test).

#### **check_correlation.csv**

This dataset contains only real background events recorded at LHCb to evaluate your submission correlation with mass locally. It contains the same columns as test.csv and mass column to check correlation with. For more details see [correlation test](https://www.kaggle.com/c/flavours-of-physics/details/correlation-test).

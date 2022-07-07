from glob import glob
from pyjet import cluster,DTYPE_EP,DTYPE_PTEPM
import numba as nb
from itertools import chain
import scipy as sp
from datetime import date
import optparse
import sys
import IPython
import awkward as ak
import vector
vector.register_awkward()
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import uproot3
import uproot
import uproot4
import numpy as np
import math
import time

gen_tree = 'FloatingpointThresholdDummyHistomaxGenmatchGenclustersntuple/HGCalTriggerNtuple'

#raw_dir = '/home/llr/cms/sarkar/DataHtoInv/0000/'
#files = glob.glob('/home/llr/cms/sarkar/DataHtoInv/0000/ntuple*.root')
#output_dir = '/home/llr/cms/sarkar/HDF5/'
branches_gen=['event','gen_pdgid','gen_phi', 'gen_eta','gen_status','gen_pt', 'gen_energy']
branches_genjet=['event','genjet_phi', 'genjet_eta','genjet_pt', 'genjet_energy']
branches_cl3d=['event','cl3d_pt','cl3d_eta','cl3d_phi','cl3d_energy']
full = ["gen_eta","gen_phi","gen_pt","gen_pdgid","gen_energy","gen_status","genjet_eta","genjet_phi","genjet_pt","genjet_energy","cl3d_pt","cl3d_energy","cl3d_eta","cl3d_phi"]

#for filename in os.listdir(raw_dir):
#print ("starting file")
#print(filename)

df =[]

tree = uproot.open(sys.argv[1])[gen_tree]
df = tree.arrays(branches_cl3d,library='pd')

indexing = tree.arrays(['event','cl3d_eta','gen_eta'])


fulljet_pt = []
fulljet_eta = []
fulljet_phi = []
fulljet_constituent = []

setidx = {i[0] for i,row  in df.iterrows() }

#for i_event in range(tree.num_entries):

for i_event in list(setidx):
   cl3d = df.loc[i_event,['cl3d_pt','cl3d_eta','cl3d_phi','cl3d_energy']]
   cl3d.columns=["pT","eta","phi","energy"]
   cl3d = cl3d.assign(mass=0)
   #go dataframe to np array, formatted for input into fastjet    
   cl3dVectors = np.array(cl3d[["pT","eta","phi","energy"]].to_records(index=False).astype([(u'pT', '<f8'), (u'eta', '<f8'), (u'phi', '<f8'), (u'mass', '<f8')]) )    
   clusterVals = cluster(cl3dVectors,R=0.4,algo="antikt")
   _jets = clusterVals.inclusive_jets() 
   ak4jet_eta = []
   ak4jet_phi = []
   ak4jet_pt = []
   ak4jetmatched_pt = []
   constituent = []
   for j,jet in enumerate(_jets):
       ak4jet_pt.append(jet.pt)
       ak4jet_eta.append(jet.eta)
       ak4jet_phi.append(jet.phi)
       constituent.append((len(_jets[j].constituents_array())))

   tmpjet_pt = ak.Array(ak4jet_pt)
   tmpjet_eta = ak.Array(ak4jet_eta)
   tmpjet_phi = ak.Array(ak4jet_phi)
   tmpjet_const = ak.Array(constituent)
   fulljet_pt.append(tmpjet_pt)
   fulljet_eta.append(tmpjet_eta)
   fulljet_phi.append(tmpjet_phi)
   fulljet_constituent.append(tmpjet_const)
   #print(tmpjet[1])
   #break
ak4jet_pt=ak.Array(fulljet_pt)
ak4jet_eta=ak.Array(fulljet_eta)
ak4jet_phi=ak.Array(fulljet_phi)
ak4jet_const=ak.Array(fulljet_constituent)



ak4jet = ak.zip({"pt":ak4jet_pt,"eta":ak4jet_eta,"phi":ak4jet_phi})
ak4jet = ak4jet[(abs(ak4jet.eta)>1.7) & (abs(ak4jet.eta)<2.8) & (ak4jet.pt>5)]

ak4jet_pd = ak.to_pandas(ak4jet)


df_gen = tree.arrays(["gen_pt", "gen_eta", "gen_phi","gen_energy","gen_pdgid","gen_status","cl3d_eta"],library='ak')
df_gentrimmed = df_gen[ak.count(df_gen['cl3d_eta'],axis=1)>0]
gen_pt = df_gentrimmed["gen_pt"]
gen_eta = df_gentrimmed["gen_eta"]
gen_phi = df_gentrimmed["gen_phi"]
gen_energy = df_gentrimmed["gen_energy"]
gen_pdgid = df_gentrimmed["gen_pdgid"]
gen_status = df_gentrimmed["gen_status"]
genzip = ak.zip({"pt":gen_pt,"eta":gen_eta,"phi":gen_phi,"energy":gen_energy,"pdgid":gen_pdgid,"status":gen_status})
gen = genzip[(genzip.status==23) & (abs(genzip.pdgid)<6) & (genzip.pt>30) & (abs(genzip.eta)>1.7) & (abs(genzip.eta)<2.8)]

gen_pd = ak.to_pandas(gen)

df_genjet = tree.arrays(["genjet_pt", "genjet_eta", "genjet_phi","genjet_energy","cl3d_eta"],library='ak')
df_genjettrimmed = df_genjet[ak.count(df_genjet['cl3d_eta'],axis=1)>0]
genjet_pt = df_genjettrimmed["genjet_pt"]
genjet_eta = df_genjettrimmed["genjet_eta"]
genjet_phi = df_genjettrimmed["genjet_phi"]
genjet_energy = df_genjettrimmed["genjet_energy"]
genjetzip = ak.zip({"pt":genjet_pt,"eta":genjet_eta,"phi":genjet_phi,"energy":genjet_energy})
genjet = genjetzip
genjet = genjetzip[(abs(genjetzip.eta)>1.7) & (genjetzip.pt>10) & (abs(genjetzip.eta)<2.8)]

genjet_pd = ak.to_pandas(genjet)

genjetmain = ak.with_name(genjet, "Momentum3D")
genmain = ak.with_name(gen, "Momentum4D")
gen_genjet = ak.cartesian({"gen": genmain, "genjet": genjetmain})
gen, genjet = ak.unzip(gen_genjet)

dRmatchedgen = ak.Array(gen.deltaR(genjet))
mindRmatchedgen = ak.flatten(ak.min(gen.deltaR(genjet),axis = -1), axis = None)

dRmatchedgen_pd = ak.to_pandas(dRmatchedgen)
mindRmatchedgen_pd = ak.to_pandas(mindRmatchedgen)

genjetmatchedzip = ak.Array(genjet[((gen.deltaR(genjet)<0.2) & ((genjet.pt/gen.pt)>0.75) & ((genjet.pt/gen.pt)<1.25))])
genmatchedzip = ak.Array(gen[((gen.deltaR(genjet)<0.2) & ((genjet.pt/gen.pt)>0.75) & ((genjet.pt/gen.pt)<1.25))])

genjetmatchedzip_pd = ak.to_pandas(genjetmatchedzip)
genmatchedzip_pd = ak.to_pandas(genmatchedzip)

genmatchedmain = ak.with_name(genmatchedzip, "Momentum3D")
genjetmatchedmain = ak.with_name(genjetmatchedzip, "Momentum3D")


ak4jets = ak.with_name(ak4jet, "Momentum3D")

genmatched_genjetmatched = ak.cartesian({"genjets": genmatchedmain, "genjetmatched": genjetmatchedmain})
genmatched, genjetmatched999 = ak.unzip(genmatched_genjetmatched)

reco_genjetmatched = ak.cartesian({"ak4jets": ak4jets, "genjetmatched": genjetmatchedmain})
ak4jet, genjetmatched = ak.unzip(reco_genjetmatched)

dRjetmatched = ak.flatten((ak4jet.deltaR(genjetmatched)), axis=None)
mindRjetmatched = ak.flatten(ak.min(ak4jet.deltaR(genjetmatched),axis = -1), axis = None)


dRjetmatched_pd = ak.to_pandas(dRjetmatched)        
mindRjetmatched_pd = ak.to_pandas(mindRjetmatched) 


ak4jetmatched = ak.Array(ak4jet[((ak4jet.deltaR(genjetmatched)<0.4) & (ak4jet.pt/genjetmatched.pt>0.5) & (ak4jet.pt/genjetmatched.pt<1.5))])
genjetmatchedfinal = ak.Array(genjetmatched[((ak4jet.deltaR(genjetmatched)<0.4) & (ak4jet.pt/genjetmatched.pt>0.5) & (ak4jet.pt/genjetmatched.pt<1.5))])

ak4jetmatched_pd = ak.to_pandas(ak4jetmatched)
genjetmatchedfinal_pd = ak.to_pandas(genjetmatchedfinal)

#filename2 = filename.replace('.root','.hdf5')

#save files to savedir in HDF
store = pd.HDFStore("jetalgo_"+sys.argv[2]+".hdf5", mode='w')
store['gen_clean'] = gen_pd
store['genjet_clean'] = genjet_pd
store['dR_matched_gen'] = dRmatchedgen_pd
store['mindR_matched_gen'] = mindRmatchedgen_pd
store['gen_matched'] = genmatchedzip_pd
store['genjet_matched'] = genjetmatchedzip_pd
store['dR_jet_matched'] = dRjetmatched_pd
store['mindR_jet_matched'] = mindRjetmatched_pd
store['genjet_matched_ak4jets'] = genjetmatchedfinal_pd
store['ak4jet'] = ak4jet_pd
store['ak4jet_matched'] = ak4jetmatched_pd
store.close()
print ("opened file")
print("jetalgo_"+sys.argv[2]+".hdf5")

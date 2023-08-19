import os
from tqdm import tqdm
import pandas as pd
import prody as prd
import torch
import esm

DATALOC = 'data/proteins'
LABELLOC = 'data/labels.txt'
ESM_BATCH_SIZE = 1

def load_proteins(dataloc=DATALOC):
    # read all proteins in .pdb files
    prd.confProDy(verbosity='none') # suppress load messages
    proteins = {}
    for file in os.listdir(dataloc):
        filename = os.fsdecode(file)
        if filename.endswith(".pdb") : 
            p = prd.parsePDB(os.path.join(dataloc, filename), verbosity=None)
            proteins[filename[:filename.find('.pdb')]] = p
    return proteins

def load_labels(labelloc=LABELLOC):
    # read protein labels
    return pd.read_csv(labelloc, sep=' ', header=None).rename(columns={0:'protein_name',1:'value'})

def proteins_dataframe(proteins):
    # create dataframe of relevant data
    # iterate over proteins and get sequence and atom coordinates w.r.t. Calpha carbons
    df = []
    for key, value in proteins.items():
        ca = value.ca
        df.append({'protein_name':key, 'coords':ca.getCoords(), 'sequence':ca.getSequence(), 'num_atoms': ca.numAtoms()})
    return pd.DataFrame(df)

def esm_embeddings(data, esm_batch_size=ESM_BATCH_SIZE):
    # get esm embeddings for proteins
    # based on README example at https://github.com/facebookresearch/esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    model = model.to('cpu')
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    token_reps = []
    with torch.no_grad():
        for bix in tqdm(range(0,len(data),esm_batch_size), desc='esm_embeddings'):
            results = model(batch_tokens[bix:bix+esm_batch_size], repr_layers=[6], return_contacts=True)
            for b in range(esm_batch_size):
                amino_str = batch_strs[bix+b]
                token_reps.append(results["representations"][6][b,1:len(amino_str)+1]) # skip 0 which is start token
    return token_reps

def load_all_data(dataloc=DATALOC, labelloc=LABELLOC, esm_batch_size=ESM_BATCH_SIZE):
    print('loading proteins...')
    proteins = load_proteins(dataloc)
    protein_df = proteins_dataframe(proteins)
    print('loading labels...')
    labels = load_labels(labelloc)
    protein_df = protein_df.merge(labels, on='protein_name')
    
    batch_data = list(protein_df[['protein_name','sequence']].itertuples(index=False, name=None))
    print('loading esm embeddings...')
    embeddings = esm_embeddings(batch_data, esm_batch_size=esm_batch_size)
    print('done.')

    return protein_df, embeddings



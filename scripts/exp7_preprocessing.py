import os
import sys
import math
import numpy as np

from bioservices import UniProt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import trange, tqdm
import pandas as pd

skip_duplicate_uniprot_ids = True

def process_data(current_directory='.'):
    
    # Fetch original data (from DeepSequence paper)
    if not os.path.exists('BLAT_ECOLX_1_b0.5.a2m'):
        os.system('wget https://sid.erda.dk/share_redirect/a5PTfl88w0/BLAT_ECOLX_1_b0.5.a2m')

    filename = "BLAT_ECOLX_1_b0.5.a2m"

    record_map = {}
    for record in SeqIO.parse(filename, "fasta"):
        if record.id == "BLAT_ECOLX/24-286":
            record.id = 'QUERY_P62593/24-286'
        key = record.id
        assert key not in record_map
        record_map[key] = record

    ids = list(record_map.keys())
    info_map = {}
    for i, batch in enumerate(tqdm(np.array_split(ids, math.ceil(len(ids)/100)))):

        # print(i)
        u = UniProt(verbose=True)
        u.settings.TIMEOUT = 240
        df = u.get_df(['id:'+id.split('/')[0].split('_')[1] for id in batch], limit=None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        result = df[['Entry', 'Sequence', 'Taxonomic lineage (PHYLUM)', 'Protein families']]

        for i in range(len(result)):
            row = result.iloc[i]
            if len(row['Sequence']) > 0:
                info_map[row['Entry']] = row.fillna('Other').to_dict()
                
    records = []
    records_full = []
    uniprot_ids_processed = {}
    for id in record_map:
        info_map_key = id.split('/')[0].split('_')[1]
        if info_map_key in info_map:

            if info_map_key in uniprot_ids_processed:
                print("WARNING - duplicate entry:", info_map_key)
                if skip_duplicate_uniprot_ids:
                    continue
            uniprot_ids_processed[info_map_key] = True
            
            info = info_map[info_map_key]
            record = record_map[id]
            record.description = "["+info['Taxonomic lineage (PHYLUM)']+"]"
            records.append(record)

            record_full = SeqRecord(
                    Seq(info['Sequence']),
                    id=info['Entry'],
                    description="["+info['Taxonomic lineage (PHYLUM)']+"]"
                )
            records_full.append(record_full)
        else:
            print("Not found: ", id, info_map_key)

    with open(os.path.join(current_directory, "BLAT_ECOLX_1_b0.5_labeled.fasta"), "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
    with open(os.path.join(current_directory, "BLAT_ECOLX_1_b0.5_full_sequences_labeled.fasta"), "w") as output_handle:
        SeqIO.write(records_full, output_handle, "fasta")

    # Alignment using clustal omega
    os.system('clustalo --verbose -i BLAT_ECOLX_1_b0.5_full_sequences_labeled.fasta -o BLAT_ECOLX_alignment_clustal0.fasta')

    # Refinement using maxalign
    os.system('perl maxalign.pl -v=1 -a BLAT_ECOLX_alignment_clustalo.fasta; mv heuristic.fsa BLAT_ECOLX_alignment_clustalo_maxalign.fasta')

    
if __name__ == '__main__':

    process_data()
    
    # Sanity check
    os.system('head -n 10 BLAT_ECOLX_1_b0.5_labeled.fasta')
    print()
    os.system('head -n 10 BLAT_ECOLX_1_b0.5_full_sequences_labeled.fasta')
    print()
    os.system('head -n 10 BLAT_ECOLX_alignment_clustal0.fasta')

import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

if __name__ == "__main__":
    data = []
    split = str(sys.argv[1])
    groups = ("euk", "bac", "vir", "arc")
    for name in groups:
        with open(f'{name}_full_exp/{split}.txt') as f:
            content = f.readlines()
        data.append(content)

    seqs = [ ]
    for i in range(len(data)):
        for j in range(len(data[i])):
            line = data[i][j]
            line = line[:-1] # remove \n
            line = line.replace(" ", "")
            record = SeqRecord(Seq(line))
            seqs.append(record)
    
    with open(f"{split}_combined.fasta", "w") as output_handle:
        SeqIO.write(seqs, output_handle, "fasta")
            
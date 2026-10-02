# Data table contents

Energy tables from the supplementary material in the publication below.

Lacroix, E., Viguera, A. R., & Serrano, L. (1998). Elucidating the folding problem of α-helices: local motifs, long-range electrostatics, ionic-strength dependence and prediction of NMR parameters. Journal of molecular biology, 284(1), 173-191. https://doi.org/10.1006/jmbi.1998.2145

The paper uses the terminology of Richardson & Richardson (1988) where STC (S, strand; T, turn; and C, coil) indicates a non-helical conformation and He is a helical residue. Python indexing is used to describe these positions in the model.
```text
Name:      N''  N'   Ncap N1   N2   N3   N4   N5.............C5   C4   C3   C2   C1   Ccap C'   C''  
Structure: STC  STC  STC -He---He---He---He---He---He---He---He---He---He---He---He---STC  STC  STC
Index:     -2   -1   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
```


## table_1_lacroix
Free energies in Kcal/mol for the intrinsic tendencies of the different amino acids to be at different positions of an a-helix.
The first column corresponds to the 20 amino acids in one letter. The N-terminus blocking acetyl or succynil groups are indicated by Ac and the C-terminus blocking amide group by Am.

Nc-1 	Normal N-cap values.
Nc-2 	N-cap values when there is a Pro at position N1.
Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.
Cc-1 	Normal C-cap values.
Cc-2 	C-cap values when there is a Pro residue at position C’.
N1	    Intrinsic helical propensities at position N1.
N2	    Intrinsic helical propensities at position N2.
N3	    Intrinsic helical propensities at position N3.
N4 	    Intrinsic helical propensities at position N4.
Ncen	Intrinsic helical propensities between N4 and C-cap. For charged residues these could change depending of the degree of ionisation.  
Neutral	Intrinsic helical propensities at positions higher than N4.

The Glu (E) row departs from the published table: its charged columns are 0.172 higher and its Neutral value is 1.046, so that Glu⁻ and Glu⁰ cost +0.572 and +0.366 kcal/mol relative to Ala, the measured helix propensities of Chakrabartty et al. (1994, Protein Sci. 3, 843; Table 2), the source the published table cites for neutral and charged propensities.

The Ile (I) and Leu (L) N3 cells depart from the published table: I 1.08 → 1.26 and L 0.98 → 1.21. The published N3 values come from Petukhov et al. (1998, J. Mol. Biol. 278, 279; peptides Ac-AAXAAAAARAAARGGY-NH2) and were fitted in an earlier version of the model. With this model's other terms they no longer reproduce those peptides. Each cell was re-fitted so that the model's X3/Ala helicity ratio equals the measured one in two independent hosts, and the mean of the two fits was taken. The fits were +0.20 (Ile) and +0.16 (Leu) for Petukhov et al. (1998), and +0.17 and +0.30 for Iqbalsyah & Doig (2004, Protein Sci. 13, 32; Ac-AAXAAAAKAAAAKAGY-NH2). Met, Val and Gly N3 cells reproduce Petukhov et al. (1998) as published and are unchanged.

Capping box: when the N-cap is Ser, Thr, Asp or Asn and N3 is Glu, the Nc-3 value gets an extra -0.9 kcal/mol, weighted by the helix-state ionisation of that Glu (set in `get_dG_Ncap`). The published Nc-3 column gives Ser no capping-box advantage at all (Nc-3 -0.65 vs Nc-1 -0.70). Peptide measurements that isolate the motif put it at -0.9 +/- 0.3 kcal/mol (mean of five within-paper comparisons): Glu vs Ala, Gln and Asp at N3 with a Ser N-cap (Zhou et al. 1994, Proteins 18, 1, Table I), and Ser vs Ala N-caps (Petukhov et al. 1996, Biochemistry 35, 387, Table 1). Glu outperforms Gln by as much as it outperforms Ala, which places the effect on the charged carboxylate.


## table_2_lacroix
Energy contributions in Kcal/mol * 100, of the interactions between different amino acids at positions N' (rows) and N4 (columns) in a hydrophobic staple motif. 

The hydrophobic staple motif is only considered whenever the N-cap residue is Asn, Asp, Ser, Pro or Thr.  The above values are multiplied by 1 in the following cases: i) whenever the N-cap residue is Asn, Asp, Ser, or Thr and the N3 residue is Glu, Asp or Gln. ii) whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr.  In all other cases they are multiplied by 0.5.

## table_3_lacroix
Energy contributions in cal/mol Kcal/mol * 100 of the interactions between the different amino acids at positions C3 (rows) and C’ (columns), in the Schellman motif. 

The Schellman motif is only considered whenever Gly is the C-cap residue.

## table_4a_lacroix
Energy contributions in cal/mol Kcal/mol * 100 of the non-charged side chain-side chain interactions between the different amino acids at positions i,i+3. 

The interaction free energies correspond to those between non-charged residues, or in the case of two residues that can be charged to those cases in which at least one of the two is non-charged (the interaction is scaled according to the population of charged and neutral forms of the participating amino acids).  

## table_4b_lacroix
Energy contributions in cal/mol Kcal/mol * 100 of the non-charged side chain-side chain interactions between the different amino acids at positions i,i+4. 

The interaction free energies correspond to those between non-charged residues, or in the case of two residues that can be charged to those cases in which at least one of the two is non-charged (the interaction is scaled according to the population of charged and neutral forms of the participating amino acids).

In this implementation (tables 4a and 4b), a pair of an acidic (Asp, Glu) and a basic (Lys, Arg, His) residue is treated differently: its value is the side-chain hydrogen bond, which is present whether or not the acid is charged and does not depend on salt (Scholtz et al. 1993, Biochemistry 32, 9668; Smith & Scholtz 1998, Biochemistry 37, 33), so it is applied in every ionisation state. The ionic part of such a pair is the Coulomb term.

## table_6_coil_lacroix
Average distance between charged groups (Å).

The distances shown in this table have been obtained from the analysis of the protein database, or from a modeled helix, as indicated in Methods and represent average values. In the different columns we show the distance between residues at position i and i+x.  The amino acid pairs are shown in one-letter code. The nomenclature for the helix position of the charged residues (columns N-cap etc...) is that of Richardson & Richardson (1988).

Rcoil 		Distance between i, i+x pairs of charged residues in the whole protein database.   
RcoilRest 		Average distance between i, i+x pairs of charged residues in the reference state 
		not included in Rcoil. 

## table_6_helix_lacroix
Average distance between charged groups (Å).

The distances shown in this table have been obtained from the analysis of the protein database, or from a modeled helix, as indicated in Methods and represent average values. In the different columns we show the distance between residues at position i and i+x.  The amino acid pairs are shown in one-letter code. The nomenclature for the helix position of the charged residues (columns N-cap etc...) is that of Richardson & Richardson (1988).

Helix		Distance between i 	i+x pairs of charged residues located inside an a-helix (excluding caps). 
Helixrest 		The same but for all possible charged pairs not included before. 
Ncap		Distance between the N-cap residue (i) and a helical residue located at position i+x.  
N’		Distance between the residue at position N’ (i) and a helical residue located at position i+x. 
Ccap		Distance between the C-cap residue (i) and a helical residue located at position i-x. 
C’		Distance between residue C’ (i) when residue C’ is not a Gly and a helical residue 
		located at position i-x. 
C’gcap		Distance between residue C’ (i) when residue C’ is a Gly and a helical residue 
		located at position i-x.  The presence of a Gly allows dihedral angles forbidden, 
		or not favourable, for other residues and therefore affects to the distance between 
		a charged group at position C’ and the helix charged residues.  
N-cap f 		Distance between the free N-terminal group when this group is located at the N-cap 
		position and a helical residue at position i+x.  
N’ f 		Distance between the free N-terminal group, when this group is located at position N’, 
		and a helical residue at position i+x. 
C-cap f 		Distance between the free C-terminal group, when this group is located at the C-cap 
		position, and a helical residue at position i-x.   
C’ f 		Distance between the free C-terminal group, when this group is located at position C’, 
		and a helical residue at position i-x.  

## table_7_Ncap_lacroix
Distances (Å) between charged amino acids and the half charge from the helix macrodipole.

The distances in Å shown in this table have been obtained from the analysis of the protein database as indicated in Methods.  The nomenclature for the helix position of the charged residues (columns N-cap etc...) is that of Richardson & Richardson (1988).

## table_7_Ccap_lacroix
Distances (Å) between charged amino acids and the half charge from the helix macrodipole.

The distances in Å shown in this table have been obtained from the analysis of the protein database as indicated in Methods.  The nomenclature for the helix position of the charged residues (columns N-cap etc...) is that of Richardson & Richardson (1988).

## table_7_coulomb_Ncap, table_7_coulomb_Ccap
The distances (Å) used by the side chain-macrodipole term for charged residues inside the helix. They are identical to tables 7 above (Lacroix 1998, supplementary Table VII) and are kept as separate files so that this term's distances can be varied independently of the flanking-residue screening, which reads tables 7.

## pka_values
The pKa values for N- and C-termini as well as ionizable side chains when incorporated in a peptide. These are reference (base) pKa values: the pKa of each group in an unstructured peptide, before the model's helix- and coil-state electrostatic shifts are applied. Rule: a value measured in an unstructured peptide at the conditions of the helicity data (0-5 C, low salt) is used where one exists, otherwise the value measured in unstructured alanine pentapeptides.

- Nterm 8.00, Cterm 3.67, His 6.54, Cys 8.55, Tyr 9.84, Lys 10.40: Thurlkill, Grimsley, Scholtz & Pace (2006) Protein Sci. 15, 1214, Table 2 (Ac-AAXAA-NH2, 0.1 M KCl, 25 C).
- Glu 4.49: no unstructured-peptide Glu pKa measured at 0-5 C and low salt was found. The value is Thurlkill's 4.25 (25 C, 0.1 M KCl) plus +0.24. That offset is the measured difference for Asp in the same peptide between the two sets of conditions: 3.91 at 0 C in 10 mM NaCl (Huyghues-Despointes et al. 1993) vs 3.67 at 25 C in 0.1 M KCl (Thurlkill et al. 2006). Both are side-chain carboxyl groups with similar ionisation enthalpies and activity-coefficient behaviour, so Asp's measured shift is transferred to Glu. Independent check: Glu pKa values measured in helical peptides at 0 C cluster at 4.48-4.61 (N-cap, N1, N2, N3 and C-cap; Doig & Baldwin 1995 Protein Sci. 4, 1325; Cochran, Penel & Doig 2001; Cochran & Doig 2001; Iqbalsyah & Doig 2004), above the 25 C value.
- Asp 3.91: Huyghues-Despointes, Scholtz & Baldwin (1993) Protein Sci. 2, 80 (Ac-AADAA-NH2 by NMR, 0 C, 10 mM NaCl). This is the same peptide Thurlkill et al. measured at 3.67, but at the conditions of the helicity data.
- Arg 13.8: Fitch, Platzer, Okon, Garcia-Moreno & McIntosh (2015) Protein Sci. 24, 752.
- Sc (succinyl) 4.5: unchanged; no measured source found.

These replace values from Nozaki and Tanford (1967), which were measured in model compounds rather than peptides.



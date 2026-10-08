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

Three hydrophobic i,i+3 cells of table 4a depart from the published table: Leu-Leu -35 -> -16, Val(i)-Leu(i+3) -20 -> -3 and Ile(i)-Leu(i+3) -26 -> -7 (kcal/mol x 100; row = residue i). The published side-chain values were derived from a mean-force potential over NMR data on protein fragments (Munoz & Serrano 1994, Nat. Struct. Biol. 1, 399), not from peptide energetics. Two independent determinations show these i,i+3 values are too strong, and each new value is their mean. (1) Measured: Padmanabhan & Baldwin (1994, Protein Sci. 3, 1992) compared i,i+3 and i,i+4 placements of the same pairs in Ac-YKAAA...K-NH2. Holding the published i,i+4 cells, their measured i,i+4 minus i,i+3 helicity differences imply i,i+3 energies of -0.18 (Leu-Leu), -0.01 (Val-Leu) and -0.01 (Ile-Leu). (2) Computed: Creamer & Rose (1995, Protein Sci. 4, 1305, Table 7) give -0.13, -0.05 and -0.13 by Boltzmann-weighted modelling. The two sources agree only when Leu is the C-terminal residue of the pair, so the Leu(i)-X(i+3) cells and all i,i+4 cells are unchanged.

In this implementation (tables 4a and 4b), a pair of an acidic (Asp, Glu) and a basic (Lys, Arg, His) residue is treated differently: its value is the side-chain hydrogen bond, which is present whether or not the acid is charged and does not depend on salt (Scholtz et al. 1993, Biochemistry 32, 9668; Smith & Scholtz 1998, Biochemistry 37, 33), so it is applied in every ionisation state. The ionic part of such a pair is the Coulomb term.

Three acid-base cells depart from the published tables: His(i)-Glu(i+3) 0 -> -18 (table 4a), His(i)-Glu(i+4) 0 -> -21 and Lys(i)-Asp(i+4) -40 -> -24 (table 4b) (kcal/mol x 100; row = residue i). They come from Smith & Scholtz (1998, Biochemistry 37, 33), one of the sources of the published Table IV. Smith & Scholtz measured Glu/Asp-Lys and Glu-His pairs in both orientations at i,i+3, i,i+4 and i,i+5 in the host Ac-AAQAAAAQAAAAQAAY-NH2, with the acid neutral and charged, at 0.01, 1.0 and 2.5 M NaCl. Their (i,i+5) peptides carry no interaction, so the difference between an (i,i+k) peptide and its (i,i+5) partner cancels the charge-macrodipole terms. Each new value is the cell value for which this model reproduces those differences, averaged over salt concentrations and ionisation states (6-9 conditions per cell, standard deviation 0.05-0.11 kcal/mol). The same procedure returns the published value for the Glu-Lys, Lys-Glu and Asp-Lys cells at i+3 and i+4 within 0.03 kcal/mol, which is the check that it is sound. The published table has no His-Glu interaction. The measured one persists when His is neutral (pH 8.5), as expected for a hydrogen bond. The data are `pyagadir/data/peptides/1998_smith_biochem37_table3.tsv`.

One further acid-base cell departs from the published table: Arg(i)-Glu(i+3) -35 -> -5 (table 4a; kcal/mol x 100). Two independent orientation-resolved datasets show the published value is too strong. Huyghues-Despointes, Klingler & Baldwin (1993, Protein Sci. 2, 80) measured Ac-AEAARAEAARAEAARY-NH2 against its reverse, Ac-ARAAEARAAEARAAEY-NH2, at pH 2.5 and 7.0 and 0.01 and 1.0 M NaCl. Meuzelaar, Vreede & Woutersen (2016, Biophys. J. 110, 2328, supplementary Table S1) measured melting temperatures of Ac-A(AEAAR)3A-NH2, Ac-A(ARAAE)3A-NH2 and Ac-A(RAAAE)3A-NH2 at pH 7.0 and 2.5. With -0.35 the model reverses the measured orientation preference at neutral pH in both: Glu-first is more helical (Huyghues-Despointes, +31 helix points at 0.01 M) and melts 19 K higher (Meuzelaar), but the model favours Arg-first. With the acid neutral, the model also over-favours Arg-first. Scanning the cell, both datasets are reproduced best near -0.05 (Huyghues-Despointes flat between -0.05 and 0; Meuzelaar minimum at -0.05). Meuzelaar's rotamer analysis (supplement section 1.2) gives the structural reason: an Arg(i)-Glu(i+3) pair has no intrinsically preferred side-chain rotamers that allow a salt bridge in the helix. The neutral-pH orientation preference remains under-predicted in both datasets at this value, so the cell does not carry the whole effect.

## table_6_coil_lacroix
Average distance between charged groups (Å).

The distances shown in this table have been obtained from the analysis of the protein database, or from a modeled helix, as indicated in Methods and represent average values. In the different columns we show the distance between residues at position i and i+x.  The amino acid pairs are shown in one-letter code. The nomenclature for the helix position of the charged residues (columns N-cap etc...) is that of Richardson & Richardson (1988).

Rcoil 		Distance between i, i+x pairs of charged residues in the whole protein database.   
RcoilRest 		Average distance between i, i+x pairs of charged residues in the reference state 
		not included in Rcoil. 

Pairs without a row of their own (His-His, and any pair with Tyr or Cys) take the RcoilRest row here and the HelixRest row of table_6_helix_lacroix, as these captions define them. The pair energy is q_i q_j W(r) at the self-consistent fractional charges, so a His pair is weighted by its protonation automatically; with the ionisation free energy it reproduces exact enumeration of the four protonation microstates of a His-His peptide (tests/test_ionization.py).

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
The distances (Å) used by the side chain-macrodipole term for charged residues inside the helix. They are identical to tables 7 above (Lacroix 1998, supplementary Table VII). The model reads these copies (`EnergyCalculator._sidechain_dipole_potential`); tables 7 are kept as the published reference.

## pka_values
The pKa values for N- and C-termini as well as ionizable side chains when incorporated in a peptide. These are reference (base) pKa values: the pKa of each group in an unstructured peptide, before the model's helix- and coil-state electrostatic shifts are applied. Rule: a value measured in an unstructured peptide at the conditions of the helicity data (0-5 C, low salt) is used where one exists, otherwise the value measured in unstructured alanine pentapeptides.

- Nterm 8.00, Cterm 3.67, His 6.54, Cys 8.55, Lys 10.40: Thurlkill, Grimsley, Scholtz & Pace (2006) Protein Sci. 15, 1214, Table 2 (Ac-AAXAA-NH2, 0.1 M KCl, 25 C).
- Tyr 9.5: Lacroix, Viguera & Serrano (1998) J. Mol. Biol. 284, 173, Table 1 measured the Tyr side-chain pKa at 9.4 +/- 0.1 in the unstructured control peptide KR-1c (CD titration, 278 K, no added salt; 9.2 and 9.3 in the helical peptides KR-1a and KR-1f). That apparent value includes the stabilisation of the phenolate by the peptide's Lys and Arg. The model's coil state puts that at about -0.1 pKa unit, so the base value is 9.5. Thurlkill et al. give 9.84 at 25 C in 0.1 M KCl.
- Glu 4.49: no unstructured-peptide Glu pKa measured at 0-5 C and low salt was found. The value is Thurlkill's 4.25 (25 C, 0.1 M KCl) plus +0.24. That offset is the measured difference for Asp in the same peptide between the two sets of conditions: 3.91 at 0 C in 10 mM NaCl (Huyghues-Despointes et al. 1993) vs 3.67 at 25 C in 0.1 M KCl (Thurlkill et al. 2006). Both are side-chain carboxyl groups with similar ionisation enthalpies and activity-coefficient behaviour, so Asp's measured shift is transferred to Glu. Independent check: Glu pKa values measured in helical peptides at 0 C cluster at 4.48-4.61 (N-cap, N1, N2, N3 and C-cap; Doig & Baldwin 1995 Protein Sci. 4, 1325; Cochran, Penel & Doig 2001; Cochran & Doig 2001; Iqbalsyah & Doig 2004), above the 25 C value.
- Asp 3.91: Huyghues-Despointes, Scholtz & Baldwin (1993) Protein Sci. 2, 1604 (Ac-AADAA-NH2 by NMR, 0 C, 10 mM NaCl). This is the same peptide Thurlkill et al. measured at 3.67, but at the conditions of the helicity data.
- Arg 13.8: Fitch, Platzer, Okon, Garcia-Moreno & McIntosh (2015) Protein Sci. 24, 752.
- Sc (succinyl) 4.5: unchanged; no measured source found.
- Nterm_Y 7.2: the alpha-amino pKa depends on the N-terminal residue (7.6-8.9 across residues in Doig & Baldwin 1995 Protein Sci. 4, 1325, Table 2), and a residue-specific row overrides Nterm for a free N-terminus starting with that residue. Lacroix, Viguera & Serrano (1998) J. Mol. Biol. 284, 173, Table 1 measured the free alpha-amino group of Tyr1 at 7.1 +/- 0.1 in the unstructured control peptide KR-1c (CD titration, 278 K, no added salt), the conditions of their pH-titration panels. That apparent value includes the repulsion from the peptide's Lys and Arg; the model's coil state puts that repulsion at about 0.1 pKa unit at these conditions, so the base value is 7.2.

These replace values from Nozaki and Tanford (1967), which were measured in model compounds rather than peptides.



## Electrostatic free energy (no table)
The ionisation states of all titratable groups are solved self-consistently in the helix and coil states (Lacroix 1998, eqs 8-11): each group's pKa is shifted by the energy of its charged state in the field of the macrodipole (helix only) and of the other groups. Two rules keep this consistent with the energy terms:

- The solver uses the same interaction model as the energy terms: the side chain-macrodipole law (nearest end by Muñoz 1995-II eq. 11, far end and flanks as half charges), the terminal-macrodipole term with its locality gate, and the terminal-side chain and side chain-side chain distances of Lacroix 1998 supplementary Table VI. A pair the energy terms do not model has the same distance in both states, so it shifts pKas alike and adds no helix-coil energy.
- The electrostatic terms charge sum q_i q_j W_ij + sum q_i phi_i at the converged charges. That is the mean-field energy; the free energy also contains the cost of moving each group's ionisation away from its intrinsic value. The segment energy therefore includes the ionisation free energy, sum over groups of -RT ln[(1 + x e^(-psi/RT)) / (1 + x)] - q psi, helix minus coil (`chemistry.ionization_free_energy`). It vanishes for groups that stay fully charged or fully neutral; for partly ionised groups the energy alone over-counts the interaction (for Asp next to Arg at pH 2.5, by a factor of 1.8). With it, the mean-field result matches exact enumeration of all protonation states to within 0.003 kcal/mol on test peptides.

## Salt terms (constants in `models.py`, no table)
Two terms describe how salt acts on the helix itself, separately from the Debye-Hückel screening of charge interactions. Both constants come from the NaCl series of Scholtz, York, Stewart & Baldwin (1991) J. Am. Chem. Soc. 113, 5102 (Figure 2): Ac-(AAQAA)3Y-NH2, 0 C, pH 7. That peptide has no charged groups, so its helicity against salt concentration isolates these two terms from everything else in the model.

- Salting-in, per helical segment: `-0.30 (1 - exp(-3 I))` kcal/mol, I = ionic strength (M). This is the shape of equation 12 of Lacroix et al. (1998), and 0.30 kcal/mol is the amplitude Scholtz et al. report for the stabilisation of the neutral helix at low salt. Lacroix et al. fitted 0.15. Their fit had no salting-out term, and their data include ~1 M salt, where the salting-out of a 16-residue helix is about 0.14 kcal/mol; that is roughly the difference between 0.30 and 0.15.
- Salting-out (Hofmeister), per helical residue (the segment without its caps): `+0.0085 I` kcal/mol. Scholtz et al. obtain the NaCl Hofmeister slope from the points above 1.2 M, where the salting-in has saturated: -0.082 kcal/mol per M for the complete helix in their Lifson-Roig analysis (sigma = 0.0029), i.e. about 0.006 per residue. The value here is the same derivation done in this model: the per-residue coefficient at which the model reproduces the measured helicity fall over those points (best 0.008-0.009). The model's helicity responds less to a uniform per-residue energy than the Lifson-Roig fit does, so the coefficient is larger than the published one.

With both terms the model reproduces the measured NaCl curve of that peptide: a rise of 5.1 helix points to the 0.5-1.2 M plateau (measured 5.6) and a fall of 7.3 points from there to 2.95 M (measured 7.2). Smith & Scholtz (1998) Biochemistry 37, 33 (Table 2) give the same-sign salting-out above 1 M in the same host peptide, with less precision.

Limitations: the coefficient is for NaCl (KCl behaves similarly). Hofmeister effects are ion-specific: Na2SO4 stabilises the helix and CaCl2 destabilises it about twice as strongly as NaCl (Scholtz et al. 1991), and the model has no salt-identity input. Both constants were measured at 0 C and are applied at all temperatures. The per-residue form assumes salting-out scales with the number of residues that change conformation.

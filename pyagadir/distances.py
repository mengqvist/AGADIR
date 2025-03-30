import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json
from pathlib import Path
from importlib.resources import files

class AminoAcid(Enum):
    """Enum for the 20 standard amino acids
    """
    ALA = "A"
    CYS = "C"
    ASP = "D"
    GLU = "E"
    PHE = "F"
    GLY = "G"
    HIS = "H"
    ILE = "I"
    LYS = "K"
    LEU = "L"
    MET = "M"
    ASN = "N"
    PRO = "P"
    GLN = "Q"
    ARG = "R"
    SER = "S"
    THR = "T"
    VAL = "V"
    TRP = "W"
    TYR = "Y"

class HelixCoilDistanceCalculator:
    """
    Calculator for sidechain-sidechain, sidechain-backbone, and 
    terminus-dipole distances in peptides with both helical and coil regions.
    All distances are in Angstroms.
    """
    _params = None # class variable to store the params

    @classmethod
    def load_params(cls):
        """Load the parameters for the AGADIR model.

        This method loads model parameters from JSON files and stores them in a class variable.
        The parameters are only loaded once and reused across instances to optimize memory
        and performance.

        Returns:
            dict: Dictionary containing the loaded parameters with the following keys:
                - helix_parameters: Parameters defining helix geometry
                - distance_parameters: Parameters for distance calculations
                - residue_parameters: Per-residue physical parameters
                - optimization_parameters: Optional parameters for model optimization

        Raises:
            ValueError: If parameters are missing for any amino acid in the AminoAcid enum
        """
        if cls._params is None:
            cls._params = {}

        # get params
        datapath = files("pyagadir.data.params")

        with open(datapath.joinpath("distance_params.json"), 'r') as f:
            params = json.loads(f.read())

        required_sections = ['helix_parameters', 'coil_parameters', 'residue_parameters', 'distance_parameters']
        for section in required_sections:
            if section not in params:
                raise ValueError(f"Missing required parameter section: {section}")

        # Helix geometry parameters
        cls._params['helix_parameters'] = params['helix_parameters']
        
        # Coil geometry parameters
        cls._params['coil_parameters'] = params['coil_parameters']
        
        # Distance calculation parameters
        cls._params['distance_parameters'] = params['distance_parameters']
        
        # Residue-specific parameters
        cls._params['residue_parameters'] = params['residue_parameters']
        
        # Optional optimization parameters if present
        cls._params['optimization_parameters'] = params.get('optimization_parameters', {})
        
        # Validate residue parameters
        for aa in AminoAcid:
            if aa.value not in cls._params['residue_parameters']:
                raise ValueError(f"Missing parameters for residue {aa.value}")

        return cls._params

    def __init__(self, sequence: str, i: int, j: int):
        """
        Initialize a calculator for computing distances in peptides with helical and coil regions.

        Args:
            sequence (str): Amino acid sequence in one-letter code.
            i (int): Zero-based starting position of helix.
            j (int): Length of helical region in number of residues.

        Raises:
            ValueError: If residue parameters are missing for any amino acid in sequence.
        """
        self.params = self.load_params()
        self.residue_params = self.params['residue_parameters']
        self.helix_params = self.params['helix_parameters']
        self.coil_params = self.params['coil_parameters']
        self.distance_params = self.params['distance_parameters']
        self.optimization_params = self.params['optimization_parameters']

        self.sequence = [AminoAcid(aa) for aa in sequence.upper()]
        self.length = len(sequence)
        self.helix_start = i
        self.helix_length = j
        self.helix_end = i + j - 1

        # Assign helix dipole points
        self.n_dipole = self.helix_start
        self.c_dipole = self.helix_end
    
    def _get_helix_backbone_coordinates(self, position: int) -> np.ndarray:
        """Get backbone coordinates for a residue position.
        
        Args:
            position: Integer index of the residue position in the sequence.
            
        Returns:
            np.ndarray: 3D coordinates (x,y,z) in Angstroms.
        """
        # For positions relative to helix
        rel_pos = position - self.helix_start
        
        angle = np.radians(self.helix_params['angle_per_residue'] * rel_pos)
        x = self.helix_params['radius'] * np.cos(angle)
        y = self.helix_params['radius'] * np.sin(angle)
        z = rel_pos * self.helix_params['rise_per_residue']
        
        return np.array([x, y, z])
    
    def _get_sidechain_coordinates(self, position: int) -> np.ndarray:
        """Get sidechain coordinates for a residue.

        Args:
            position: Integer index of the residue position in the sequence.

        Returns:
            np.ndarray: 3D coordinates (x,y,z) in Angstroms.
        """
        backbone_coords = self._get_helix_backbone_coordinates(position)
        
        # Get residue-specific extension
        aa = self.sequence[position].value
        extension = self.residue_params[aa]['extension']
        
        # Adjust extension based on whether residue is in helix
        if self.helix_start <= position <= self.helix_end:
            extension *= self.distance_params['helix_scaling_factor']
        else:
            extension *= self.distance_params['coil_scaling_factor']
        
        # Calculate sidechain position
        rel_pos = position - self.helix_start
        angle = np.radians(self.helix_params['angle_per_residue'] * rel_pos)
        
        x_extend = extension * np.cos(angle)
        y_extend = extension * np.sin(angle)
        
        return backbone_coords + np.array([x_extend, y_extend, 0])
        
    def _estimate_coil_distance(self, pos1: int, pos2: int) -> float:
        """Estimate distance between positions in coil regions using a worm-like chain model.
        
        Args:
            pos1, pos2: Integer positions in the sequence

        Returns:
            float: Estimated distance in Angstroms
        """
        N = abs(pos2 - pos1)
        contour_length = self.coil_params['bb_peptide_bond'] + (N + 1) * self.coil_params['bb_ca_distance']

        # Approximate random walk behavior with persistence length scaling
        return np.sqrt(2 * self.coil_params['persistence_length'] * contour_length)

    # def estimate_coil_distance(self, pos1: int, pos2: int) -> float:
    #     """
    #     Estimate distance between positions in coil regions using worm-like chain model,
    #     accounting for residue properties.
    #     """
    #     sequence_separation = abs(pos2 - pos1)
        
    #     # Calculate effective contour length considering residue properties
    #     contour_length = 0
    #     start, end = min(pos1, pos2), max(pos1, pos2)
    #     for pos in range(start, end + 1):
    #         residue = self.sequence[pos]
    #         props = self.RESIDUE_PROPERTIES[residue]
    #         # Adjust local persistence based on residue properties
    #         contour_length += self.COIL_CA_DISTANCE * (1 + 0.2 * props.flexibility)
        
    #     # Using modified worm-like chain behavior
    #     if contour_length < self.COIL_PERSISTENCE_LENGTH:
    #         return contour_length
    #     else:
    #         # Average flexibility of intervening residues affects scaling
    #         avg_flexibility = np.mean([self.RESIDUE_PROPERTIES[self.sequence[pos]].flexibility 
    #                                 for pos in range(start, end + 1)])
    #         return np.sqrt(2 * self.COIL_PERSISTENCE_LENGTH * contour_length) * (1 + 0.1 * avg_flexibility)

    def get_sidechain_sidechain_distances(self) -> np.ndarray:
        """Calculate all pairwise sidechain-sidechain distances between residues.

        Returns:
            np.ndarray: A symmetric matrix of shape (sequence_length, sequence_length) containing
                the Euclidean distances between all pairs of sidechain coordinates in Angstroms.
                The diagonal elements are 0 since they represent the same residue.
        """
        distances = np.zeros((self.length, self.length))
        
        for i in range(self.length):
            coords_i = self._get_sidechain_coordinates(i)
            for j in range(i+1, self.length):
                coords_j = self._get_sidechain_coordinates(j)
                dist = np.linalg.norm(coords_j - coords_i)
                distances[i,j] = distances[j,i] = dist
                
        return distances
    
    def get_sidechain_backbone_distances(self) -> np.ndarray:
        """Calculate all pairwise sidechain-backbone distances between residues.

        Returns:
            np.ndarray: A symmetric matrix of shape (sequence_length, sequence_length) containing
                the Euclidean distances between each residue's sidechain coordinates and every
                other residue's backbone coordinates in Angstroms.
        """
        distances = np.zeros((self.length, self.length))
        
        for i in range(self.length):
            coords_i = self._get_sidechain_coordinates(i)
            for j in range(self.length):
                coords_j = self._get_helix_backbone_coordinates(j)
                dist = np.linalg.norm(coords_j - coords_i)
                distances[i,j] = dist
                
        return distances
    
    def get_termini_dipole_distances(self) -> Tuple[List[float], List[float]]:
        """Calculate distances from each terminus to the helix dipole points.

        The distances are calculated between the backbone coordinates of the first/last 5 residues
        and the helix dipole points located at the N- and C-termini of the helix.

        Returns:
            tuple[list[float], list[float]]: A tuple containing:
                - n_term_distance: Distance from N-terminal residue to N-terminal dipole point in Angstroms
                - c_term_distance: Distance from C-terminal residue to C-terminal dipole point in Angstroms
        """
        # Calculate for the N-terminal residue
        n_term_distance = self._estimate_coil_distance(0, self.n_dipole)

        # Calculate for the C-terminal residue 
        c_term_distance = self._estimate_coil_distance(self.length-1, self.c_dipole)

        return n_term_distance, c_term_distance

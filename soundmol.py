import sys

sys.path.append(
    "/Users/uzbit/Documents/projects/soundmol/psi4conda/lib/python3.10/site-packages"
)
import psi4
from rdkit import Chem
from rdkit.Chem import AllChem
import sounddevice as sd
import numpy as np
import threading
import soundfile as sf


def get_freq_info(scf_wfn):
    frequencies = scf_wfn.frequencies().to_array()
    # Print the vibrational frequencies and their modes
    for i, freq in enumerate(frequencies):
        print(f"Frequency: {freq:.2f} cm^-1")
        print(f"Mode {i+1}:")
        print()
    return frequencies


def add_tone(signal, frequency, duration=10.0, amplitude=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    s = amplitude * np.sin(2 * np.pi * frequency * t)
    if signal.size > 0:
        signal += s
    else:
        signal = s
    return signal


def play_sound(signal, sample_rate=44100):
    sd.play(signal, samplerate=sample_rate)
    sd.wait()


def save_sound(file, signal, sample_rate=44100):
    sf.write(file, signal, sample_rate, subtype="FLOAT")


def play_tone(frequency, duration=10.0, amplitude=0.5, sample_rate=44100):
    """Generate a tone of a given frequency and play it."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    sd.play(signal, samplerate=sample_rate)
    sd.wait()


def play_freqs(freqs):
    threads = list()
    for freq in freqs:
        thread = threading.Thread(target=play_tone, args=(freq,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


smiles = "CN(C)CCC1=CNC2=C1C(=CC=C2)OP(=O)(O)O"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

# Convert RDKit molecule to XYZ format
xyz = Chem.MolToXYZBlock(mol)
print("Molecule starting geometry:")
print(xyz)

psi4.set_num_threads(10)
# psi4.set_options({'temperature': 300}) # Degrees K
mol = psi4.geometry(xyz)

# Define your molecule here.
# Ge -0.500000   0.000000   0.000000
# mol = psi4.geometry(
#     """
# 0 1
# Si  0.000000  -0.667578  -2.124659
# Ge   0.000000   0.667578  -2.124659

# H   0.923621  -1.232253  -2.126185
# H  -0.923621  -1.232253  -2.126185
# H  -0.923621   1.232253  -2.126185
# H   0.923621   1.232253  -2.126185
# """
# )

psi4.set_options({"reference": "rhf"})
psi4.optimize("scf/cc-pvdz", molecule=mol)
optimized_geometry = mol.to_string(dtype="xyz")
print("Optimized geometry:")
print(optimized_geometry)
with open("mol.xyz", "w") as fh:
    fh.write(optimized_geometry)

scf_e, scf_wfn = psi4.frequency("scf/cc-pvdz", molecule=mol, return_wfn=True)
# Get optimized geometry
freqs = get_freq_info(scf_wfn)
# play_freqs(freqs)
sound = np.array([])
for freq in freqs:
    sound = add_tone(sound, freq)
play_sound(sound)
print("Saving sound and geometry to mol.wav, mol.xyz respectively...")
save_sound("mol.wav", sound)

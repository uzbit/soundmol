import os
import sys

sys.path.append(f"{os.environ['PSI4']}/lib/python3.10/site-packages")
import psi4
from rdkit import Chem
from rdkit.Chem import AllChem
import sounddevice as sd
import numpy as np
import threading
import soundfile as sf
import matplotlib.pyplot as plt

DURATION = 10  # seconds
SAMPLE_RATE = 60000  # Hz
DO_PLOT = True


def compute_geom_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Convert RDKit molecule to XYZ format
    xyz = Chem.MolToXYZBlock(mol)
    print("Molecule starting geometry:")
    print(xyz)

    mol = psi4.geometry(xyz)
    return mol


def optimize_geom(basis, mol):
    # "scf/cc-pvdz" is more expensive/accurate basis
    print("Optimizing geometry...")
    psi4.optimize(basis, molecule=mol)
    optimized_geometry = mol.to_string(dtype="xyz")
    print("Optimized geometry:")
    print(optimized_geometry)
    with open("mol.xyz", "w") as fh:
        fh.write(optimized_geometry)
    return mol


def get_freq_info(wfn):
    frequencies = wfn.frequencies().to_array()
    # Print the vibrational frequencies and their modes
    for i, freq in enumerate(frequencies):
        print(f"Frequency: {freq:.2f} cm^-1")
        print(f"Mode {i+1}:")
        print()
    return frequencies


def compute_freqs(basis, mol):
    print("Computing normal modes...")
    scf_e, scf_wfn = psi4.frequency(basis, molecule=mol, return_wfn=True)
    freqs = get_freq_info(scf_wfn)
    return freqs
    # Get optimized geometry


def get_time_series():
    return np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)


def add_tone(signal, frequency, amplitude=0.5, phase=0.0):
    t = get_time_series()
    s = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if signal.size > 0:
        signal += s
    else:
        signal = s
    return signal


def play_sound(signal):
    sd.play(signal)
    sd.wait()


def save_sound(file, signal):
    sf.write(file, signal, SAMPLE_RATE, subtype="FLOAT")


def play_tone(frequency, amplitude=0.5):
    t = get_time_series()
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    sd.play(signal, samplerate=SAMPLE_RATE)
    sd.wait()


def play_freqs(freqs):
    threads = list()
    for freq in freqs:
        thread = threading.Thread(target=play_tone, args=(freq,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


def compute_sound(freqs):
    freqs *= 0.1  # scale all into lower freqs, cause it sounds cooler in that bass
    sound = np.array([])
    for i, freq in enumerate(freqs):
        sound = add_tone(sound, freq, phase=(2*np.pi*i/len(freqs)))
    return sound


def main(freqs):
    psi4.set_num_threads(10)
    psi4.set_options({"reference": "rhf"})

    # Compute new optimized geometry and normal mode freqs
    if not freqs:
        # Can define your molecule here.
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

        mol = psi4.geometry(
            """
        0 1
        O   0.000000   0.667578  -2.124659
        H  -0.923621   1.232253  -2.126185
        H   0.923621   1.232253  -2.126185
        """
        )

        # Or in smiles format for molecular descriptions.
        # smiles = "CN(C)CCC1=CNC2=C1C(=CC=C2)OP(=O)(O)O" # psilocybin
        # smiles = "c1ccccc1" # benzene
        # smiles = "C1=NC2=C(N1)C(=O)N=C(N2)N" # guanine
        # mol = compute_geom_from_smiles(smiles)

        basis = "scf/6-31G*"
        mol = optimize_geom(basis, mol)
        freqs = compute_freqs(basis, mol)

    sound = compute_sound(np.array(freqs))
    if DO_PLOT:
        t = get_time_series()
        plt.plot(t, sound)
        plt.show()
    play_sound(sound)
    save_sound("mol.wav", sound)


if __name__ == "__main__":
    # fmt: off
    freqs = [float(x) for x in 
        # This list is from line "post-proj  all modes:" in previous a optimization/freq calc
        # useful for not repeating the full optimized geom and freq calcs
        
        # [ # psilocybin 
        # '24.9650', '26.9743', '38.8128', '54.8156', '63.8747', '98.5076', '128.6125', '170.3233',
        # '178.7650', '189.7175', '207.5222', '237.4644', '242.1500', '263.1299',
        # '264.2467', '283.4913', '297.8675', '333.2050', '349.4775', '405.1296',
        # '411.1513', '437.9763', '448.1684', '460.7197', '486.3579', '498.6746',
        # '519.8833', '567.9706', '572.6526', '599.4034', '655.5579', '673.7376',
        # '703.2018', '741.4414', '805.8574', '829.5424', '836.9544', '875.4090',
        # '889.0397', '908.2982', '933.8316', '944.8993', '995.8904', '1006.5716',
        # '1029.9973', '1092.6482', '1093.7111', '1106.3295', '1119.0839', '1147.1656',
        # '1153.6500', '1165.1235', '1169.9776', '1171.6072', '1188.3366', '1216.3867',
        # '1226.9051', '1253.1221', '1271.8754', '1296.3278', '1357.1452', '1369.0724',
        # '1392.7282', '1406.5535', '1419.6862', '1446.3427', '1452.7451', '1458.9739',
        # '1500.3573', '1509.5064', '1549.8574', '1579.1960', '1597.0503', '1603.9517',
        # '1618.5829', '1631.3418', '1636.5279', '1645.5873', '1651.2391', '1658.8773',
        # '1671.9116', '1682.3570', '1745.3707', '1784.1004', '1822.3883', '3127.0369',
        # '3136.1681', '3215.2533', '3219.2813', '3230.7019', '3237.3553', '3245.7867',
        # '3270.9525', '3277.8953', '3295.7129', '3365.9581', '3385.8351', '3416.0938',
        # '3434.9811', '3923.6069', '4084.5121', '4090.6257'
        # ]

        # [ # benzene
        # '453.3194', '453.3406', '665.6223', '665.6535', '764.4075', '776.1351', '961.0145',
        # '961.2038', '1084.0435', '1096.6319', '1099.3562', '1099.4825', '1134.7352',
        # '1142.1497', '1142.1994', '1197.3181', '1294.3318', '1294.4026', '1351.6836',
        # '1508.2146', '1652.2056', '1652.2696', '1797.4393', '1797.5876', '3348.8239',
        # '3359.5498', '3359.8902', '3377.9902', '3378.3263', '3390.0340',
        # ]

        [ # GeH4Si 
        '381.2832', '521.1897', '586.1100', '602.1458', '620.2192', '896.9800',
        '1009.3790', '2281.3203', '2286.5841', '2427.8412', '2448.4340'
        ]

        # [ # Guanine
        # '126.4037', '163.7690', '238.4487', '315.7484', '355.5971', '373.1031', '378.9519',
        # '479.6096', '523.2115', '562.6846', '584.8192', '653.4299', '686.1117',
        # '705.3142', '752.6982', '785.3204', '823.0109', '857.2351', '915.9872',
        # '1007.5890', '1049.3600', '1106.6054', '1184.1462', '1226.2298', '1293.2379',
        # '1316.4317', '1390.3609', '1421.9610', '1506.0851', '1556.1552', '1612.6196',
        # '1649.9686', '1710.7174', '1782.6963', '1815.4482', '1856.6894', '1963.3266',
        # '3459.4160', '3805.7791', '3879.6625', '3908.0774', '3918.0339'
        # ]
    ]
    # fmt: on

    # Comment/uncomment to perform new geom/freq calculation, otherwise freqs above will be used.
    # freqs = []
    main(freqs)


# G1~G70 https://medium.com/toshiba-sbm/benchmarking-the-max-cut-problem-on-the-simulated-bifurcation-machine-e26e1127c0b0
# G72 G77 G81 https://arxiv.org/pdf/2505.18508
# G63 https://arxiv.org/pdf/2510.21105
best_known = {
    "g1": 11624,
    "g2": 11620,
    "g3": 11622,
    "g4": 11646,
    "g5": 11631,
    "g6": 2178,
    "g7": 2006,
    "g8": 2005,
    "g9": 2054,
    "g10": 2000,
    "g11": 564,
    "g12": 556,
    "g13": 582,
    "g14": 3064,
    "g15": 3050,
    "g16": 3052,
    "g17": 3047,
    "g18": 992,
    "g19": 906,
    "g20": 941,
    "g21": 931,
    "g22": 13359,
    "g23": 13344,
    "g24": 13337,
    "g25": 13340,
    "g26": 13328,
    "g27": 3341,
    "g28": 3298,
    "g29": 3405,
    "g30": 3413,
    "g31": 3310,
    "g32": 1410,
    "g33": 1382,
    "g34": 1384,
    "g35": 7687,
    "g36": 7680,
    "g37": 7691,
    "g38": 7688,
    "g39": 2408,
    "g40": 2400,
    "g41": 2405,
    "g42": 2481,
    "g43": 6660,
    "g44": 6650,
    "g45": 6654,
    "g46": 6649,
    "g47": 6657,
    "g48": 6000,
    "g49": 6000,
    "g50": 5880,
    "g51": 3848,
    "g52": 3851,
    "g53": 3850,
    "g54": 3852,
    "g55": 10299,
    "g56": 4017,
    "g57": 3494,
    "g58": 19293,
    "g59": 6086,
    "g60": 14188,
    "g61": 5796,
    "g62": 4870,
    "g63": 27047,
    "g64": 8751,
    "g65": 5562,
    "g66": 6364,
    "g67": 6950,
    "g70": 9591,
    "g72": 7008,
    "g77": 9940,
    "g81": 14060,

    # --- Spin Glass (3D Lattice) Instances ---
    # Source [1]: Festa, P. et al. (2002). "Randomized heuristics for the MAX-CUT problem". 
    # Source [2]: Myklebust, T. (2015). "SOLVING MAXIMUM CUT PROBLEMS BY SIMULATED ANNEALING".
    # Source [3]: Boros, E. et al. (2008). "A max-flow approach to improved lower bounds for QUBO".
    # Values updated to the best found in [3] (Discrete Optimization 5, Table 6) which are >= earlier benchmarks.
    
    # sg3dl 1000 Nodes (Boros 2008 Lower Bounds)
    "sg3dl101000": 896,
    "sg3dl102000": 900,
    "sg3dl103000": 892,
    "sg3dl104000": 898,
    "sg3dl105000": 886,
    "sg3dl106000": 888,
    "sg3dl107000": 900,
    "sg3dl108000": 882,
    "sg3dl109000": 902,
    "sg3dl1010000": 894, # Our best is 894, matching [2] and Boros [3]
    
    # sg3dl 2744 Nodes (Boros 2008 Lower Bounds)
    "sg3dl141000": 2446,
    "sg3dl142000": 2458,
    "sg3dl143000": 2442,
    "sg3dl144000": 2450,
    "sg3dl145000": 2446,
    "sg3dl146000": 2450,
    "sg3dl147000": 2444,
    "sg3dl148000": 2448, # Updated to GES record (was 2446)
    "sg3dl149000": 2426, # Updated to GES record (was 2424)
    "sg3dl1410000": 2458,

    # --- Ising Torus Instances ---
    # Source: 7th DIMACS Implementation Challenge.
    # Referenced in Festa et al. (2002) Figure 10.
    # Updated values from Myklebust (2015) "SOLVING MAXIMUM CUT PROBLEMS BY SIMULATED ANNEALING" (Figure 4)
    # Further updated from Boros et al. (2008) "A max-flow approach to improved lower bounds for QUBO" (Table 7a)
    "tourspm3-8-50": 458,    # Explicit filename
    "tourspm3-15-50": 3016,  # Boros et al. (2008) Table 7a (listed as toruspm3-15-50)
    "toursg3-8": 41684814,   # Boros et al. (2008) Table 7a (matches)
    "toursg3-15": 285790637, # Boros et al. (2008) Table 7a (listed as torusg3-15)

    "g000985": 2800,
    "g000035": 100,

    # Image Segmentation Instances (MQLib 2018 Benchmark)
    # Source: https://github.com/MQLib/MQLib
    # Methodology: Best of 37 heuristics (2018)
    "imgseg_100075": 10830689.4562772,
    "imgseg_100080": 3555215.83234028,
    "imgseg_100098": 23828987.3315444,
    "imgseg_103041": 8254999.81516674,
    "imgseg_105019": 7381459.36458549,
    "imgseg_106025": 2927960.45664959,
    "imgseg_108041": 12658292.9443592,
    "imgseg_108073": 7872540.40216396,
    "imgseg_112082": 20220908.0863843,
    "imgseg_117054": 19386054.2216626,
    "imgseg_118020": 11496820.063439,
    "imgseg_118035": 5328288.83899635,
    "imgseg_12003": 8860297.80108277,
    "imgseg_126039": 5895002.67006896,
    "imgseg_135037": 7550697.35299979,
    "imgseg_138032": 23346928.9128988,
    "imgseg_140055": 9943000.28170922,
    "imgseg_140075": 7675540.46399852,
    "imgseg_147021": 12023549.7631136,
    "imgseg_147062": 30359234.6678711,
}
import os
def get_best(data:str) -> int:
    key = data.split(os.sep)[-1].split(".")[0]
    return best_known.get(key, None)

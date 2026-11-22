import sys, time, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.append(sys.argv[1])
from skimage import data
import fast_color_clustering as ref
from fast_feature_clustering import FastFeatureClustering
for name in ["astronaut", "coffee", "chelsea"]:
    img = getattr(data, name)()[..., :3].copy()
    for k in (8, 16):
        t=time.perf_counter(); r = ref.FastColorClustering(n_colors=k, bits=4).fit(img); tr=time.perf_counter()-t
        g = FastFeatureClustering(n_clusters=k, bits=4, post_process_refine=False).fit(img)
        same_pal = np.array_equal(r.palette_, np.clip(np.rint(g.tree_palette_),0,255).astype(np.uint8))
        same_lab = np.array_equal(r.predict(img), g.predict(img))
        print(f"{name:9s} K={k:2d} bits={g.active_bits_} cells={g.n_initial_cells_:4d} palette_equal={same_pal} labels_equal={same_lab} ref {tr*1e3:.0f} ms / gen {g.timings_['total_s']*1e3:.0f} ms")

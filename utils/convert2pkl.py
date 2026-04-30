import argparse, json, pickle, shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_table(path):
    return pd.read_csv(path, sep=r"\s+|,", engine="python")


def one_population(h5, root):
    g = h5[root]
    names = list(g.keys())
    if len(names) != 1:
        print(f"{root} populations:", names)
    return g[names[0]]


def load_nodes(nodes_h5):
    with h5py.File(nodes_h5, "r") as f:
        pop = one_population(f, "nodes")
        node_id = pop["node_id"][()]
        node_type_id = pop["node_type_id"][()]
    return node_id, node_type_id


def find_prop(pop, edge_types, edge_type_id, group_id, group_index, candidates, default=None):
    for name in candidates:
        if name in edge_types.columns:
            mp = edge_types.set_index("edge_type_id")[name].to_dict()
            return np.array([mp[int(x)] for x in edge_type_id])

        if name in pop and isinstance(pop[name], h5py.Dataset):
            return pop[name][()]

        if group_id is not None:
            out = np.empty(edge_type_id.shape[0], dtype=np.float64)
            found = False
            for gid in np.unique(group_id):
                gid_s = str(int(gid))
                if gid_s in pop and name in pop[gid_s]:
                    sel = group_id == gid
                    out[sel] = pop[gid_s][name][()][group_index[sel]]
                    found = True
            if found:
                return out

    if default is not None:
        return np.full(edge_type_id.shape[0], default)
    raise KeyError(f"Cannot find property from {candidates}")


def remap_sources(source_node_id, source_ids):
    max_id = int(max(source_node_id.max(), source_ids.max()))
    mp = np.full(max_id + 1, -1, dtype=np.int64)
    mp[source_ids.astype(np.int64)] = np.arange(len(source_ids), dtype=np.int64)
    out = mp[source_node_id.astype(np.int64)]
    keep = out >= 0
    return out, keep


def build_edges(edges_h5, edge_types_csv, source_ids=None):
    edge_types = read_table(edge_types_csv)

    with h5py.File(edges_h5, "r") as f:
        pop = one_population(f, "edges")

        src = pop["source_node_id"][()].astype(np.int64)
        trg = pop["target_node_id"][()].astype(np.int64)
        etype = pop["edge_type_id"][()].astype(np.int64)

        gid = pop["edge_group_id"][()].astype(np.int64) if "edge_group_id" in pop else None
        gidx = pop["edge_group_index"][()].astype(np.int64) if "edge_group_index" in pop else None

        receptor = find_prop(pop, edge_types, etype, gid, gidx,
                             ["receptor_type", "receptor_type_id", "receptor"], default=1).astype(np.int64)
        delay = find_prop(pop, edge_types, etype, gid, gidx,
                          ["delay", "syn_delay", "delay_ms"], default=1.0).astype(np.float32)
        weight = find_prop(pop, edge_types, etype, gid, gidx,
                           ["syn_weight", "weight", "weight_mean", "weight_max"]).astype(np.float32)

    if source_ids is not None:
        src, keep = remap_sources(src, source_ids)
        trg, etype, receptor, delay, weight = trg[keep], etype[keep], receptor[keep], delay[keep], weight[keep]

    out = []
    key_df = pd.DataFrame({"etype": etype, "r": receptor, "d": delay})
    keys = key_df.drop_duplicates().to_numpy()

    for et, r, d in keys:
        sel = (etype == et) & (receptor == r) & (delay == d)
        out.append({
            "source": src[sel].astype(np.int64),
            "target": trg[sel].astype(np.int64),
            "params": {
                "receptor_type": int(r),
                "weight": weight[sel].astype(np.float32),
                "delay": float(d),
            },
        })
    return out


def load_json_if_exists(row, components_dir):
    name = row.get("dynamics_params", None)
    if name is None or pd.isna(name):
        return {}

    name = str(name)
    components_dir = Path(components_dir)

    candidates = [
        components_dir / "cell_models" / "nest_models" / name,
        components_dir / "cell_models" / "aibs_models" / name,
        components_dir / name,
    ]

    for p in candidates:
        if p.exists():
            print(f"loading dynamics params: {p}")
            with open(p) as f:
                return json.load(f)

    matches = list(components_dir.rglob(Path(name).name))
    if matches:
        with open(matches[0]) as f:
            return json.load(f)

    raise FileNotFoundError(f"Cannot find {name} under {components_dir}")



def get_any(d, names, default=None):
    for n in names:
        if n not in d:
            continue

        v = d[n]
        if v is None:
            continue

        if isinstance(v, str) and v.strip() == "":
            continue

        try:
            na = pd.isna(v)
            if np.asarray(na).all():
                continue
        except Exception:
            pass

        return v

    return default



def as2(x, default=0.0):
    if x is None:
        return np.array([default, default], dtype=np.float32)
    a = np.array(x, dtype=np.float32).reshape(-1)
    if a.size == 1:
        a = np.array([a[0], a[0]], dtype=np.float32)
    return a[:2].astype(np.float32)


def as4(x, default=5.5):
    if x is None:
        return np.array([default, default, default, default], dtype=np.float32)
    a = np.array(x, dtype=np.float32).reshape(-1)
    if a.size < 4:
        a = np.pad(a, (0, 4 - a.size), mode="edge")
    return a[:4].astype(np.float32)


def node_params_from_row(row, components_dir):
    d = dict(row)
    d.update(load_json_if_exists(row, components_dir))

    # Best case: Allen/NEST-style dynamics json already contains these.
    E_L = get_any(d, ["E_L", "El"])
    V_th = get_any(d, ["V_th", "th_inf", "init_threshold"])
    V_reset = get_any(d, ["V_reset", "V_m", "init_voltage"])
    C_m = get_any(d, ["C_m", "C"])
    g = get_any(d, ["g", "g_m"])

    if g is None and get_any(d, ["R_input"]) is not None:
        g = 1.0 / float(get_any(d, ["R_input"]))

    tau_syn = as4(get_any(d, ["tau_syn", "tau_syns"]))
    t_ref = float(get_any(d, ["t_ref", "spike_cut_length"], 2.0))

    if get_any(d, ["asc_decay"]) is not None:
        # nest_models 里的 asc_decay 已经是 Chen 公式 exp(-dt * k) 需要的 k
        k = as2(get_any(d, ["asc_decay"]), default=0.0)
    elif get_any(d, ["asc_tau_array", "asc_tau"]) is not None:
        tau = np.array(get_any(d, ["asc_tau_array", "asc_tau"]), dtype=np.float32).reshape(-1)[:2]
        if np.nanmax(tau) < 10:
            tau = tau * 1000.0
        k = 1.0 / np.maximum(tau, 1e-6)
    else:
        k = as2(get_any(d, ["k"]), default=0.0)


    asc_amps = as2(get_any(d, ["asc_amps", "asc_amp_array"]), default=0.0)

    missing = [k for k, v in {
        "E_L": E_L, "V_th": V_th, "V_reset": V_reset, "C_m": C_m, "g": g
    }.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing node params {missing}. Check dynamics_params for row:\n{row}")

    return {
        "V_th": np.float32(V_th),
        "g": np.float32(g),
        "E_L": np.float32(E_L),
        "k": k.astype(np.float32),
        "C_m": np.float32(C_m),
        "V_reset": np.float32(V_reset),
        "tau_syn": tau_syn.astype(np.float32),
        "t_ref": np.float32(t_ref),
        "asc_amps": asc_amps.astype(np.float32),
    }


def build_network_dat(network_dir, components_dir, out_dir):
    node_ids, node_type_ids = load_nodes(network_dir / "v1_nodes.h5")
    node_types = read_table(network_dir / "v1_node_types.csv")

    nodes = []
    for _, row in node_types.iterrows():
        ntid = int(row["node_type_id"])
        ids = node_ids[node_type_ids == ntid].astype(np.int64)
        nodes.append({"ids": ids, "params": node_params_from_row(row, components_dir)})

    edges = build_edges(network_dir / "v1_v1_edges.h5", network_dir / "v1_v1_edge_types.csv")

    with open(out_dir / "network_dat.pkl", "wb") as f:
        pickle.dump({"nodes": nodes, "edges": edges}, f, protocol=4)


def empty_spikes(n):
    return [np.array([], dtype=np.float32) for _ in range(n)]


def build_input_dat(network_dir, out_dir):
    lgn_ids, _ = load_nodes(network_dir / "lgn_nodes.h5")
    lgn_edges = build_edges(network_dir / "lgn_v1_edges.h5", network_dir / "lgn_v1_edge_types.csv", source_ids=lgn_ids)

    inputs = [
        [{"ids": np.arange(len(lgn_ids), dtype=np.int64), "spikes": empty_spikes(len(lgn_ids))}, lgn_edges]
    ]

    bkg_edges_h5 = network_dir / "bkg_v1_edges.h5"
    bkg_types_csv = network_dir / "bkg_v1_edge_types.csv"
    bkg_nodes_h5 = network_dir / "bkg_nodes.h5"

    if bkg_edges_h5.exists() and bkg_types_csv.exists() and bkg_nodes_h5.exists():
        bkg_ids, _ = load_nodes(bkg_nodes_h5)
        bkg_edges = build_edges(bkg_edges_h5, bkg_types_csv, source_ids=bkg_ids)
        inputs.append([{"ids": np.arange(len(bkg_ids), dtype=np.int64), "spikes": empty_spikes(len(bkg_ids))}, bkg_edges])
    else:
        inputs.append([{"ids": np.arange(1, dtype=np.int64), "spikes": empty_spikes(1)}, []])

    with open(out_dir / "input_dat.pkl", "wb") as f:
        pickle.dump(inputs, f, protocol=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network-dir", required=True, help="Allen simulations/v1_point/network directory")
    ap.add_argument("--components-dir", required=True, help="Allen simulations/v1_point/components directory")
    ap.add_argument("--out", required=True, help="Output GLIF_network directory")
    args = ap.parse_args()

    network_dir = Path(args.network_dir)
    components_dir = Path(args.components_dir)
    out_dir = Path(args.out)
    (out_dir / "network").mkdir(parents=True, exist_ok=True)

    shutil.copy2(network_dir / "v1_nodes.h5", out_dir / "network" / "v1_nodes.h5")
    shutil.copy2(network_dir / "v1_node_types.csv", out_dir / "network" / "v1_node_types.csv")

    build_network_dat(network_dir, components_dir, out_dir)
    build_input_dat(network_dir, out_dir)

    print("Wrote:")
    print(out_dir / "network_dat.pkl")
    print(out_dir / "input_dat.pkl")
    print(out_dir / "network" / "v1_nodes.h5")
    print(out_dir / "network" / "v1_node_types.csv")


if __name__ == "__main__":
    main()









import argparse, json, re, shutil
from pathlib import Path

PAT = re.compile(r"(?P<pid>\d+)_c(?P<cam>\d+)")

def read_idx(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        parts = line.split()
        rel = parts[0].replace('\\', '/')
        m = PAT.search(Path(rel).name)
        if not m: raise ValueError(f"cannot parse pid/camera: {rel}")
        rows.append((rel, int(m['pid']), int(m['cam'])))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=Path('/sda1/XXX/data/LLCM'))
    ap.add_argument('--out', type=Path, default=Path('/sda1/XXX/data/llcm_v2'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(); raw, out = a.raw, a.out
    if not (raw/'idx').is_dir(): raise SystemExit(f'missing {raw}/idx')
    if out.exists() and not a.force: raise SystemExit(f'{out} exists; use --force')
    if out.exists(): shutil.rmtree(out)
    (out/'images').mkdir(parents=True)
    rows = {k: read_idx(raw/'idx'/f'{k}.txt') for k in ('train_vis','train_nir','test_vis','test_nir')}
    identities = {}
    def add(kind, rel, pid, cam):
        
        cam0 = (cam-1) if kind.endswith('vis') else (cam-1+9)
        name = f'{pid:08d}_{cam0:02d}_{len(identities.setdefault(pid, {}).setdefault(cam0, [])):04d}.jpg'
        identities[pid][cam0].append(name)
        shutil.copy2(raw/rel, out/'images'/name)
        return name
    train_pids=set(); query=[]; gallery=[]
    for k in ('train_vis','train_nir'):
        for rel,pid,cam in rows[k]: train_pids.add(pid); add(k,rel,pid,cam)
    for rel,pid,cam in rows['test_vis']: query.append(add('test_vis',rel,pid,cam))
    for rel,pid,cam in rows['test_nir']: gallery.append(add('test_nir',rel,pid,cam))
    maxpid=max(identities); ncam=max(max(x) for x in identities.values())+1
    ident=[[] for _ in range(maxpid+1)]
    for pid, cams in identities.items():
        ident[pid]=[cams.get(c,[]) for c in range(ncam)]
    test_pids={int(x) for x in (raw/'idx'/'test_id.txt').read_text().replace(',',' ').split()}
    assert test_pids == {int(Path(x).name.split('_')[0]) for x,_,_ in rows['test_vis']}
    meta={'name':'LLCM','shot':'multiple','num_cameras':ncam,'identities':ident,
          'query_fnames':query,'gallery_fnames':gallery}
    (out/'meta.json').write_text(json.dumps(meta, indent=2))
    (out/'splits.json').write_text(json.dumps([{'trainval':sorted(train_pids),'query':sorted(test_pids),'gallery':sorted(test_pids)}], indent=2))
    print(f'converted {len(train_pids)} train IDs, {len(query)} queries, {len(gallery)} gallery images -> {out}')
if __name__ == '__main__': main()

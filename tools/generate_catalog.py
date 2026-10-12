from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import os
import re
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
INDEX_PATH = ROOT / "index.json"

ASSETS = [
    {"id":"blue-ceramic-pool-tile-001","slug":"blue-ceramic-pool-tile","name":"Blue Ceramic Pool Tile","prefix":"BlueCeramicPoolTile","category":"Tiles","tags":["ceramic","pool","blue","glossy","liminal"],"description":"Glossy blue ceramic pool tiles with subtle per-tile color variation and aged grout.","generator":"tile","tile_color":[43,141,166],"grout_color":[136,150,148],"roughness":0.20,"seed":101,"featured":True},
    {"id":"dark-ceramic-tile-001","slug":"dark-ceramic-tile","name":"Dark Ceramic Tile","prefix":"DarkCeramicTile","category":"Tiles","tags":["ceramic","tile","dark","glossy","architectural"],"description":"Dark charcoal ceramic tiles with restrained gloss, fine grout and subtle tonal variation.","generator":"tile","tile_color":[48,52,54],"grout_color":[94,96,94],"roughness":0.24,"seed":113},
    {"id":"concrete-wall-001","slug":"concrete-wall-001","name":"Concrete Wall 001","prefix":"ConcreteWall001","category":"Concrete","tags":["concrete","wall","brutalist","architectural"],"description":"Neutral cast concrete with fine pores, broad tonal breakup and subtle surface waviness.","generator":"concrete","seed":211,"featured":True},
    {"id":"wet-concrete-floor-001","slug":"wet-concrete-floor","name":"Wet Concrete Floor","prefix":"WetConcreteFloor","category":"Ground","tags":["concrete","wet","floor","reflective","liminal"],"description":"Dark concrete floor with irregular wet patches, shallow surface variation and low roughness puddling.","generator":"wet_concrete","seed":223,"featured":True},
    {"id":"rusty-metal-panel-001","slug":"rusty-metal-panel","name":"Rusty Metal Panel","prefix":"RustyMetalPanel","category":"Metal","tags":["metal","rust","industrial","panel"],"description":"Industrial steel panels with seams, worn oxidation, pitting and a physically based metallic mask.","generator":"metal","seed":307,"featured":True},
    {"id":"old-plaster-wall-001","slug":"old-plaster-wall","name":"Old Plaster Wall","prefix":"OldPlasterWall","category":"Plaster","tags":["plaster","wall","aged","cracked"],"description":"Warm aged plaster with chalky variation, worn discoloration and fine branching cracks.","generator":"plaster","seed":401},
    {"id":"mossy-stone-001","slug":"mossy-stone","name":"Mossy Stone","prefix":"MossyStone","category":"Stone","tags":["stone","moss","damp","nature"],"description":"Weathered stone blocks with damp tonal variation and moss gathering around recessed joints.","generator":"mossy_stone","seed":503},
    {"id":"wood-planks-001","slug":"wood-planks-001","name":"Wood Planks 001","prefix":"WoodPlanks001","category":"Wood","tags":["wood","planks","floor","architectural"],"description":"Warm medium-brown wood planks with directional grain, fine seams and restrained surface relief.","generator":"wood","seed":601},
    {"id":"checker-tile-001","slug":"checker-tile","name":"Checker Tile","prefix":"CheckerTile","category":"Tiles","tags":["tile","checker","floor","liminal"],"description":"Muted checker floor tiles with lightly aged ceramic faces and recessed grout.","generator":"checker","seed":701},
]

def smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0.0, 1.0)
    return t*t*(3.0-2.0*t)

def periodic_noise(u, v, seed, layers=5):
    out = np.zeros_like(u, dtype=np.float32)
    weight_sum = 0.0
    for i in range(layers):
        k = seed + i*97
        fx, fy = 2+(k*3)%17, 3+(k*5)%19
        fx2, fy2 = 2+(k*7)%23, 2+(k*11)%17
        phase = ((k*0.61803398875)%1.0)*2.0*np.pi
        w = 0.58**i
        out += w*(0.62*np.sin(2*np.pi*(fx*u+fy*v)+phase)+0.38*np.cos(2*np.pi*(fx2*u-fy2*v)+phase*0.73))
        weight_sum += w
    return np.clip(out/max(weight_sum,1e-6), -1.0, 1.0)

def coords(size):
    axis = np.arange(size,dtype=np.float32)/np.float32(size)
    return np.meshgrid(axis,axis)

def normal_from_height(height, strength=18.0):
    dx=(np.roll(height,-1,1)-np.roll(height,1,1))*0.5
    dy=(np.roll(height,-1,0)-np.roll(height,1,0))*0.5
    nx,ny,nz=-dx*strength,dy*strength,np.ones_like(height,dtype=np.float32)
    inv=1.0/np.sqrt(nx*nx+ny*ny+nz*nz)
    n=np.stack([nx*inv,ny*inv,nz*inv],-1)
    return np.clip((n*0.5+0.5)*255.0,0,255).astype(np.uint8)

def finalize(base,height,roughness,ao,metallic=None,normal_strength=18.0):
    r={"BaseColor":(np.clip(base,0,255).astype(np.uint8),"RGB"),"Normal":(normal_from_height(height.astype(np.float32),normal_strength),"RGB"),"Roughness":((np.clip(roughness,0,1)*255).astype(np.uint8),"L"),"Height":((np.clip(height,0,1)*255).astype(np.uint8),"L"),"AO":((np.clip(ao,0,1)*255).astype(np.uint8),"L")}
    if metallic is not None:r["Metallic"]=((np.clip(metallic,0,1)*255).astype(np.uint8),"L")
    return r

def gen_tile(size,cfg,checker=False):
    u,v=coords(size); tiles=6 if checker else 8
    gx,gy=u*tiles,v*tiles; fx,fy=gx-np.floor(gx),gy-np.floor(gy)
    d=np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)); profile=smoothstep(0.034,0.080,d).astype(np.float32); edge=1-profile
    n1=periodic_noise(u,v,cfg["seed"],4); n2=periodic_noise((u+0.193)%1,(v+0.327)%1,cfg["seed"]+17,4)
    tx=np.floor(gx).astype(np.int16)%tiles; ty=np.floor(gy).astype(np.int16)%tiles
    var=(((tx*11+ty*7+tx*ty*5+cfg["seed"])%19)/18.0-0.5).astype(np.float32)
    grout=np.array(cfg.get("grout_color",[155,155,150]),dtype=np.float32)
    if checker:
        a=np.array([207,201,188],dtype=np.float32); b=np.array([83,92,96],dtype=np.float32); chk=((tx+ty)%2).astype(np.float32)
        tile_rgb=a[None,None,:]*(1-chk[...,None])+b[None,None,:]*chk[...,None]
        base=grout[None,None,:]*(1-profile[...,None])+tile_rgb*profile[...,None]
    else:
        tile=np.array(cfg["tile_color"],dtype=np.float32); base=grout[None,None,:]*(1-profile[...,None])+tile[None,None,:]*profile[...,None]
    base += profile[...,None]*(7*var[...,None]+4*n1[...,None])
    mineral=np.clip((n2-0.30)*0.13,0,0.09)*(0.25+0.75*edge); stain=np.array([125,110,88],dtype=np.float32)
    base=base*(1-mineral[...,None])+stain[None,None,:]*mineral[...,None]
    height=np.clip(0.12+0.71*profile+profile*(0.018*var+0.011*n1),0,1)
    rough=np.clip(cfg.get("roughness",0.24)+0.44*edge+0.07*(n2+1)*0.5,0.12,0.92)
    ao=np.clip(1.0-0.50*edge-0.04*(1-smoothstep(0.06,0.18,d)),0.25,1.0)
    return finalize(base,height,rough,ao,normal_strength=21.0)

def gen_concrete(size,cfg,wet=False):
    u,v=coords(size); n1=periodic_noise(u,v,cfg["seed"],6); n2=periodic_noise((u+0.271)%1,(v+0.137)%1,cfg["seed"]+31,5); micro=periodic_noise((u+0.071)%1,(v+0.411)%1,cfg["seed"]+83,4)
    pores=smoothstep(0.48,0.76,periodic_noise(u,v,cfg["seed"]+151,5)); height=np.clip(0.50+0.055*n1+0.022*n2-0.035*pores,0,1)
    gray=150+18*n1+7*micro-16*pores; base=np.stack([gray*1.01,gray,gray*0.98],-1)
    if wet:
        wetmask=smoothstep(0.05,0.52,periodic_noise((u+0.33)%1,(v+0.19)%1,cfg["seed"]+211,5))*(0.40+0.60*smoothstep(-0.55,0.55,n2))
        base *= (0.57+0.20*(1-wetmask))[...,None]; rough=np.clip(0.76-0.58*wetmask+0.08*(micro+1)*0.5,0.12,0.88); height-=0.010*wetmask; ao=np.clip(0.92-0.10*pores-0.05*wetmask,0.62,1)
    else:
        rough=np.clip(0.74+0.12*(micro+1)*0.5,0.62,0.94); ao=np.clip(0.94-0.13*pores,0.66,1)
    return finalize(base,height,rough,ao,normal_strength=15.0)

def gen_metal(size,cfg):
    u,v=coords(size); p=3; gx,gy=u*p,v*p; fx,fy=gx-np.floor(gx),gy-np.floor(gy); d=np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)); seam=1-smoothstep(0.018,0.055,d)
    n1=periodic_noise(u,v,cfg["seed"],6); n2=periodic_noise((u+0.29)%1,(v+0.17)%1,cfg["seed"]+29,6); rust=smoothstep(0.05,0.58,0.62*n1+0.38*n2+0.55*seam); pitting=smoothstep(0.52,0.78,periodic_noise(u,v,cfg["seed"]+113,5))*rust
    steel=np.array([82,88,89],dtype=np.float32); rr=np.array([151,71,32],dtype=np.float32); base=steel[None,None,:]*(1-rust[...,None])+rr[None,None,:]*rust[...,None]; base+=(8*n2-12*pitting)[...,None]
    height=np.clip(0.54-0.14*seam-0.055*pitting+0.012*n2,0,1); rough=np.clip(0.34+0.52*rust+0.08*pitting,0.24,0.95); ao=np.clip(0.95-0.30*seam-0.16*pitting,0.48,1); metallic=np.clip(0.92*(1-rust)+0.08*rust,0,1)
    return finalize(base,height,rough,ao,metallic,20.0)

def gen_plaster(size,cfg):
    u,v=coords(size); n1=periodic_noise(u,v,cfg["seed"],6); n2=periodic_noise((u+0.13)%1,(v+0.37)%1,cfg["seed"]+41,5); warp=0.08*n1
    l1=np.abs(np.sin(2*np.pi*(2*u+3*v+warp))); l2=np.abs(np.sin(2*np.pi*(5*u-2*v-0.7*warp)+1.4)); cracks=np.clip((0.035-np.minimum(l1,l2))/0.035,0,1)*smoothstep(-0.25,0.65,n2)
    height=np.clip(0.55+0.045*n1+0.020*n2-0.085*cracks,0,1); warm=205+18*n1+7*n2; base=np.stack([warm*1.02,warm*0.99,warm*0.93],-1); aged=smoothstep(0.22,0.72,n2); base[...,0]-=10*aged; base[...,1]-=13*aged; base[...,2]-=16*aged; base-=(38*cracks)[...,None]
    rough=np.clip(0.76+0.13*(n2+1)*0.5+0.06*cracks,0.66,0.96); ao=np.clip(0.97-0.34*cracks,0.52,1)
    return finalize(base,height,rough,ao,normal_strength=17.0)

def gen_mossy_stone(size,cfg):
    u,v=coords(size); cols,rows=5,4; gx,gy=u*cols,v*rows; fx,fy=gx-np.floor(gx),gy-np.floor(gy); d=np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)); profile=smoothstep(0.028,0.085,d); edge=1-profile
    n1=periodic_noise(u,v,cfg["seed"],6); n2=periodic_noise((u+0.17)%1,(v+0.23)%1,cfg["seed"]+57,5); moss=smoothstep(0.08,0.58,0.58*n1+0.42*n2+0.75*edge)
    stone=np.array([132,132,119],dtype=np.float32); grout=np.array([76,78,70],dtype=np.float32); base=grout[None,None,:]*edge[...,None]+stone[None,None,:]*profile[...,None]+(14*n1)[...,None]*profile[...,None]; mr=np.array([62,85,35],dtype=np.float32); base=base*(1-0.72*moss[...,None])+mr[None,None,:]*(0.72*moss[...,None])
    height=np.clip(0.18+0.64*profile+0.03*n1*profile+0.035*moss,0,1); rough=np.clip(0.74+0.13*edge+0.10*moss,0.68,0.98); ao=np.clip(0.98-0.50*edge-0.12*moss,0.28,1)
    return finalize(base,height,rough,ao,normal_strength=19.0)

def gen_wood(size,cfg):
    u,v=coords(size); planks=6; gx=u*planks; fx=gx-np.floor(gx); d=np.minimum(fx,1-fx); profile=smoothstep(0.020,0.060,d); seam=1-profile; n1=periodic_noise(u,v,cfg["seed"],5); n2=periodic_noise((u+0.41)%1,(v+0.07)%1,cfg["seed"]+61,4)
    grain=(0.5+0.5*np.sin(2*np.pi*(22*v+1.7*n1+0.25*np.sin(2*np.pi*3*u)))-0.5)*2.0; pid=np.floor(gx).astype(np.int16)%planks; pv=(((pid*11+cfg["seed"])%13)/12.0-0.5).astype(np.float32)
    brown=np.array([137,91,55],dtype=np.float32); dark=np.array([61,43,31],dtype=np.float32); base=dark[None,None,:]*seam[...,None]+brown[None,None,:]*profile[...,None]+profile[...,None]*(18*grain+12*pv+5*n2)[...,None]
    height=np.clip(0.22+0.58*profile+profile*(0.025*grain+0.012*n2),0,1); rough=np.clip(0.55+0.18*(n2+1)*0.5+0.16*seam,0.42,0.88); ao=np.clip(0.98-0.46*seam,0.40,1)
    return finalize(base,height,rough,ao,normal_strength=16.0)

GENERATORS={"tile":lambda s,c:gen_tile(s,c,False),"checker":lambda s,c:gen_tile(s,{**c,"grout_color":[145,143,135],"roughness":0.33},True),"concrete":lambda s,c:gen_concrete(s,c,False),"wet_concrete":lambda s,c:gen_concrete(s,c,True),"metal":gen_metal,"plaster":gen_plaster,"mossy_stone":gen_mossy_stone,"wood":gen_wood}
MAP_KEYS={"BaseColor":"baseColor","Normal":"normal","Roughness":"roughness","Height":"height","AO":"ao","Metallic":"metallic"}
MAP_NAMES={"BaseColor":"Base Color","Normal":"Normal","Roughness":"Roughness","Height":"Height","AO":"AO","Metallic":"Metallic"}

def parse_resolutions():
    out=[]
    for part in os.environ.get("TAFC_RESOLUTIONS","1024:1K,2048:2K,4096:4K").split(","):
        size,label=part.split(":",1); out.append((int(size),label))
    return out

def generate_asset(cfg,resolutions):
    asset_dir=ASSETS_DIR/cfg["id"]; asset_dir.mkdir(parents=True,exist_ok=True); dl=asset_dir/"downloads"; dl.mkdir(exist_ok=True)
    max_size,max_label=max(resolutions,key=lambda x:x[0]); maps=GENERATORS[cfg["generator"]](max_size,cfg); max_paths={}; folder=asset_dir/max_label; folder.mkdir(exist_ok=True)
    for map_name,(array,mode) in maps.items():
        p=folder/f'{cfg["prefix"]}_{max_label}_{map_name}.png'; Image.fromarray(array,mode=mode).save(p,compress_level=1); max_paths[map_name]=p
    files={max_label:list(max_paths.values())}
    for size,label in resolutions:
        if label==max_label:continue
        target=asset_dir/label; target.mkdir(exist_ok=True); files[label]=[]
        for map_name,source in max_paths.items():
            with Image.open(source) as im:
                p=target/f'{cfg["prefix"]}_{label}_{map_name}.png'; im.resize((size,size),Image.Resampling.LANCZOS).save(p,compress_level=1); files[label].append(p)
    low_size,low_label=min(resolutions,key=lambda x:x[0]); base=next(p for p in files[low_label] if p.name.endswith("_BaseColor.png"))
    with Image.open(base) as im: im.convert("RGB").resize((512,512),Image.Resampling.LANCZOS).save(asset_dir/"preview.webp","WEBP",quality=90,method=4)
    zips={}
    for size,label in resolutions:
        stem=re.sub(r"[^A-Za-z0-9]+","-",cfg["name"]).strip("-"); stem=re.sub(r"-001$","",stem); zp=dl/f'{stem}-001-{label}.zip'
        with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for f in files[label]:z.write(f,arcname=f.name)
            z.writestr("README.txt",f'TAfC — {cfg["name"]}\nMaps: {", ".join(MAP_NAMES[n] for n in maps)}\nResolution: {label}\nSeamless / tileable.\n')
        zips[label]=zp
    map_base=f'assets/{cfg["id"]}/{low_label}'; map_urls={MAP_KEYS[n]:f'{map_base}/{cfg["prefix"]}_{low_label}_{n}.png' for n in maps}
    meta={"id":cfg["id"],"slug":cfg["slug"],"name":cfg["name"],"category":cfg["category"],"tags":cfg["tags"],"description":cfg["description"],"seamless":True,"featured":bool(cfg.get("featured",False)),"license":"TAfC-Free","preview":f'assets/{cfg["id"]}/preview.webp',"maps":map_urls,"resolutions":[]}
    for size,label in resolutions:
        z=zips[label]; meta["resolutions"].append({"label":label,"width":size,"height":size,"download":f'assets/{cfg["id"]}/downloads/{z.name}',"sizeBytes":z.stat().st_size})
    (asset_dir/"README.md").write_text(f'# {cfg["name"]}\n\n{cfg["description"]}\n\nMaps: {", ".join(MAP_NAMES[n] for n in maps)}.\nResolutions: {", ".join(l for _,l in resolutions)}.\n',encoding="utf-8")
    del maps; gc.collect(); return meta

def main():
    resolutions=parse_resolutions(); ASSETS_DIR.mkdir(exist_ok=True); catalog=json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {"schemaVersion":1,"assets":[]}
    replace={c["id"] for c in ASSETS}; kept=[a for a in catalog.get("assets",[]) if a.get("id") not in replace]; generated=[]
    for i,cfg in enumerate(ASSETS,1): print(f'[{i}/{len(ASSETS)}] Generating {cfg["name"]}...',flush=True); generated.append(generate_asset(cfg,resolutions))
    all_assets=kept+generated; all_assets.sort(key=lambda a:(not bool(a.get("featured")),a.get("category",""),a.get("name","")))
    INDEX_PATH.write_text(json.dumps({"schemaVersion":1,"generatedAt":datetime.now(timezone.utc).isoformat(),"assets":all_assets},indent=2)+"\n",encoding="utf-8")
    print(f'Catalog now contains {len(all_assets)} assets.')

if __name__=="__main__":main()

"""Build ONE self-contained, password-protected review page (with a drawing tool) from the candidate layers.

The page is a single .html file. The images are inside it, encrypted (AES-256-GCM, key derived from your
password with PBKDF2-SHA256). Nothing is uploaded anywhere: the page makes no network requests (it also carries a
Content-Security-Policy that forbids them).

The expert sees, for each candidate: the cell picture, the existing annotations (green) and the proposed outline
(yellow), each switchable. They answer Approve / Reject and can DRAW outlines with the mouse, finger or pen.

Usage (password is asked on the keyboard, never put on the command line):
    python build_review_page.py --cycle-dir . --out ~/review_cycle0.html

Every build uses a new random salt, so a rebuilt page starts with a clean slate (earlier test clicks do not carry over).
Send the HTML through an approved channel (THWS Cloud) and the password through a DIFFERENT channel.
Do not commit the HTML file: it contains the confidential images.
"""
import argparse, base64, csv, getpass, io, json, os, secrets
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cycle-dir", default=".")
ap.add_argument("--layers-dir", default="thumbnails_layers", help="folder with Cxxx_base.png, Cxxx_gt.png, Cxxx_cand.png")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="CellFoundry candidate review (cycle 0)")
ap.add_argument("--iterations", type=int, default=300_000)
ap.add_argument("--jpeg-quality", type=int, default=90)
ap.add_argument("--limit", type=int, default=None, help="keep only the first N candidates")
ap.add_argument("--sample", type=int, default=None, help="keep N candidates spread evenly over the whole ranking (good for a small test copy)")
a = ap.parse_args()

password = os.environ.get("REVIEW_PASSWORD") or getpass.getpass("Password for the review page: ")
if len(password) < 12:
    raise SystemExit("Use a password of at least 12 characters (a passphrase of several words is best).")
if not os.environ.get("REVIEW_PASSWORD"):
    if getpass.getpass("Repeat password: ") != password:
        raise SystemExit("Passwords differ.")

cyc = Path(a.cycle_dir)
lay = cyc / a.layers_dir
rows = list(csv.DictReader(open(cyc / "review.csv", encoding="utf-8")))
if a.sample:
    step = max(1, len(rows) // a.sample)
    rows = rows[::step][: a.sample]
if a.limit:
    rows = rows[: a.limit]


def b64file(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


items = []
for r in rows:
    cid = r["candidate_id"]
    buf = io.BytesIO()
    Image.open(lay / f"{cid}_base.png").convert("RGB").save(buf, "JPEG", quality=a.jpeg_quality)
    polyf = lay / f"{cid}_poly.json"
    item = {"id": cid, "image": r["image"], "base": base64.b64encode(buf.getvalue()).decode(),
            "gt": b64file(lay / f"{cid}_gt.png"),
            "cx": int(r.get("crop_x", 0) or 0), "cy": int(r.get("crop_y", 0) or 0), "size": 256, "scale": 2}
    if polyf.exists():
        item["poly"] = json.load(open(polyf))          # editable yellow outline (canvas coordinates)
    else:
        item["cand"] = b64file(lay / f"{cid}_cand.png")  # fallback: flat picture, cannot be nudged
    items.append(item)
payload = json.dumps({"title": a.title, "items": items}).encode("utf-8")

salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
key = PBKDF2HMAC(hashes.SHA256(), 32, salt, a.iterations).derive(password.encode("utf-8"))
ct = AESGCM(key).encrypt(iv, payload, None)

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candidate review</title>
<style>
:root{--bg:#f6f5f2;--fg:#1c1c1c;--card:#fff;--mut:#6b6b6b;--ok:#1f8a4c;--no:#c0392b;--line:#dcdad4;--acc:#0b7285}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--card:#222;--mut:#a0a0a0;--line:#383838;--acc:#3bc9db}}
*{box-sizing:border-box}body{margin:0;font:16px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:1.15rem;margin:0 0 4px}.mut{color:var(--mut);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-top:12px}
input[type=password]{width:100%;padding:10px;font-size:1rem;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
button{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.ok{background:var(--ok);color:#fff;border-color:var(--ok)}button.no{background:var(--no);color:#fff;border-color:var(--no)}
button:disabled{opacity:.5;cursor:default}.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;align-items:center}.grow{flex:1;min-width:120px}
canvas{display:block;margin:0 auto;width:100%;max-width:640px;height:auto;border-radius:6px;touch-action:none;cursor:crosshair;background:#888}
.bar{height:8px;background:var(--line);border-radius:5px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--ok);width:0}
.leg b{display:inline-block;width:14px;height:4px;vertical-align:middle;margin:0 4px 0 10px}
#msg{color:var(--no);margin-top:8px;min-height:1.2em}.tag{font-weight:600}label{display:inline-flex;gap:6px;align-items:center;font-size:.92rem}
button.danger{color:var(--no)}
</style></head><body><main>
<section id="lock" class="card"><h1>Candidate review</h1>
<p class="mut">This file is confidential and stays on your computer. Enter the password you were given.</p>
<input id="pw" type="password" placeholder="Password" autocomplete="off">
<div class="row"><button id="unlock" class="ok">Open</button></div><div id="msg"></div></section>
<section id="app" hidden>
<h1 id="title"></h1>
<div class="card"><div id="prog" class="mut"></div><div class="bar"><i id="fill"></i></div>
<p class="mut leg" style="margin:10px 0 0">Is the <b style="background:#ffd700"></b><span class="tag">yellow outlined cell</span> a fat cell?
<b style="background:#00ff00"></b>Green lines = cells already annotated. <b style="background:#00e5ff"></b>Blue = your own drawing.</p>
<p class="mut" style="margin:6px 0 0"><b>Approve</b> = yes, a fat cell. <b>Reject</b> = no.
If the yellow outline does not cover the whole cell, choose <b>Nudge the yellow outline</b>, press near the yellow line and <b>drag it</b> to where the cell edge is
(like pushing clay; the "Nudge size" slider sets how much of the line moves). Then press Approve: your reshaped outline replaces the yellow one.
Or choose <b>Draw a new outline</b> and go round a cell (also for fat cells nobody marked, even when you press Reject).</p></div>
<div class="card"><div id="meta" class="mut"></div><canvas id="cv" width="512" height="512"></canvas>
<div class="row"><b style="font-size:.92rem">Tool:</b>
<label><input type="radio" name="mode" value="nudge" checked> Nudge the yellow outline (N)</label>
<label><input type="radio" name="mode" value="draw"> Draw a new outline (N)</label>
<label>Nudge size <input type="range" id="sz" min="10" max="90" value="30"></label>
<button id="rst">Reset yellow outline</button></div>
<div class="row"><label><input type="checkbox" id="sGt" checked> existing annotations</label>
<label><input type="checkbox" id="sCand" checked> yellow outline</label>
<button id="undo">Undo my last drawing (U)</button><button id="clr">Clear my drawings</button><span id="nshp" class="mut"></span></div>
<div class="row"><button id="ok" class="ok grow">Approve (A)</button><button id="no" class="no grow">Reject (R)</button></div>
<div class="row"><button id="back">&larr; Back (B)</button><button id="skip">Skip (S)</button></div></div>
<div class="card"><div class="mut">Your answers and drawings are saved in this browser automatically. When you are done:</div>
<div class="row"><button id="csv">Download decisions (CSV)</button><button id="json">Download outlines (JSON)</button><button id="folders">Save everything into folders</button></div>
<div class="mut" style="margin-top:8px">Please send back <b>decisions.csv</b> and <b>edits.json</b>. The folder option works in Chrome and Edge and creates approve / reject folders for you.</div>
<div class="row"><button id="reset" class="danger">Clear all my answers (start over)</button></div></div>
</section></main>
<script>
const ENC={salt:"__SALT__",iv:"__IV__",iter:__ITER__,ct:"__CT__"};
const b64=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u};
async function decrypt(pw){const e=new TextEncoder();const km=await crypto.subtle.importKey("raw",e.encode(pw),"PBKDF2",false,["deriveKey"]);
 const key=await crypto.subtle.deriveKey({name:"PBKDF2",salt:b64(ENC.salt),iterations:ENC.iter,hash:"SHA-256"},km,{name:"AES-GCM",length:256},false,["decrypt"]);
 const pt=await crypto.subtle.decrypt({name:"AES-GCM",iv:b64(ENC.iv)},key,b64(ENC.ct));return JSON.parse(new TextDecoder().decode(pt))}
let items=[],dec={},shp={},adj={},idx=0,cur=null,grab=null,live=null,imgs={};const $=id=>document.getElementById(id);const SK="cfreview_"+ENC.salt;
const cv=$("cv"),cx=cv.getContext("2d");
const safe=s=>s.replace(/[^A-Za-z0-9._-]+/g,"_");const fname=(it,ext)=>it.id+"__"+safe(it.image)+(ext||".jpg");
function load(){try{const o=JSON.parse(localStorage.getItem(SK)||"{}");dec=o.dec||{};shp=o.shp||{};adj=o.adj||{}}catch(e){dec={};shp={};adj={}}}
function save(){try{localStorage.setItem(SK,JSON.stringify({dec:dec,shp:shp,adj:adj}))}catch(e){}}
const mode=()=>document.querySelector("input[name=mode]:checked").value;
const polyOf=it=>live&&grab&&grab.id===it.id?live:(adj[it.id]||it.poly||null);
function firstUndecided(){const i=items.findIndex(x=>!dec[x.id]);return i<0?0:i}
const loadImg=src=>new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=src});
async function ensure(it){if(imgs[it.id])return imgs[it.id];imgs[it.id]={base:await loadImg("data:image/jpeg;base64,"+it.base),gt:await loadImg("data:image/png;base64,"+it.gt),cand:it.cand?await loadImg("data:image/png;base64,"+it.cand):null};return imgs[it.id]}
function paint(target,it,withCursor){const c=imgs[it.id];target.drawImage(c.base,0,0,512,512);
 if($("sGt").checked)target.drawImage(c.gt,0,0,512,512);
 if($("sCand").checked){const pl=polyOf(it);if(pl){target.lineWidth=3;target.strokeStyle="#ffd700";target.beginPath();pl.forEach((p,i)=>i?target.lineTo(p[0],p[1]):target.moveTo(p[0],p[1]));target.closePath();target.stroke();
  if(withCursor&&mode()==="nudge"){target.fillStyle="#ffd700";for(const p of pl){target.beginPath();target.arc(p[0],p[1],2.5,0,6.283);target.fill()}}}else if(c.cand)target.drawImage(c.cand,0,0,512,512)}
 target.lineWidth=2.5;target.strokeStyle="#00e5ff";target.fillStyle="rgba(0,229,255,.18)";
 for(const s of (shp[it.id]||[])){target.beginPath();s.forEach((p,i)=>i?target.lineTo(p[0],p[1]):target.moveTo(p[0],p[1]));target.closePath();target.fill();target.stroke()}
 if(withCursor&&cur&&cur.length>1){target.beginPath();cur.forEach((p,i)=>i?target.lineTo(p[0],p[1]):target.moveTo(p[0],p[1]));target.stroke()}}
async function render(){const it=items[idx];await ensure(it);paint(cx,it,true);const n=Object.keys(dec).length;
 $("prog").textContent=n+" of "+items.length+" decided  |  candidate "+(idx+1)+" of "+items.length+(dec[it.id]?"  |  your answer: "+dec[it.id].d:"");
 $("fill").style.width=(100*n/items.length)+"%";$("meta").textContent=it.id+"  |  from image: "+it.image;
 const k=(shp[it.id]||[]).length;$("nshp").textContent=[adj[it.id]?"yellow outline reshaped":"",k?(k+" drawing"+(k>1?"s":"")):""].filter(Boolean).join(", ")}
function decide(d){const it=items[idx];dec[it.id]={d:d,t:new Date().toISOString()};save();if(idx<items.length-1)idx++;render()}
function pos(e){const r=cv.getBoundingClientRect();return [Math.round((e.clientX-r.left)*cv.width/r.width*10)/10,Math.round((e.clientY-r.top)*cv.height/r.height*10)/10]}
const clamp=v=>Math.max(0,Math.min(512,v));
cv.addEventListener("pointerdown",e=>{if(e.button>0)return;const it=items[idx];e.preventDefault();
 if(mode()==="nudge"){const pl=polyOf(it);if(!pl){$("nshp").textContent="this candidate has no editable outline";return}
  const p0=pos(e),sg=+$("sz").value;if(Math.min(...pl.map(v=>Math.hypot(v[0]-p0[0],v[1]-p0[1])))>3*sg)return;
  cv.setPointerCapture(e.pointerId);const w=pl.map(v=>Math.exp(-((v[0]-p0[0])**2+(v[1]-p0[1])**2)/(2*sg*sg)));
  grab={id:it.id,last:p0,w:w,orig:pl.map(v=>v.slice())};live=pl.map(v=>v.slice())}
 else{cv.setPointerCapture(e.pointerId);cur=[pos(e)]}});
cv.addEventListener("pointermove",e=>{if(grab){const p=pos(e),d=[p[0]-grab.last[0],p[1]-grab.last[1]];grab.last=p;
  live.forEach((v,i)=>{v[0]=clamp(v[0]+grab.w[i]*d[0]);v[1]=clamp(v[1]+grab.w[i]*d[1])});paint(cx,items[idx],true);return}
 if(!cur)return;const p=pos(e),l=cur[cur.length-1];if(Math.hypot(p[0]-l[0],p[1]-l[1])>=3){cur.push(p);paint(cx,items[idx],true)}});
const endStroke=e=>{if(grab){const it=items[idx];adj[grab.id]=live.map(v=>[Math.round(v[0]*10)/10,Math.round(v[1]*10)/10]);grab=null;live=null;save();render();return}
 if(!cur)return;const s=cur;cur=null;if(s.length>=8){const it=items[idx];(shp[it.id]=shp[it.id]||[]).push(s);save()}render()};
cv.addEventListener("pointerup",endStroke);cv.addEventListener("pointercancel",()=>{cur=null;grab=null;live=null;render()});
$("rst").onclick=()=>{const it=items[idx];if(adj[it.id]){delete adj[it.id];save();render()}};
function undo(){const it=items[idx];if((shp[it.id]||[]).length){shp[it.id].pop();if(!shp[it.id].length)delete shp[it.id];save();render()}}
$("undo").onclick=undo;$("clr").onclick=()=>{const it=items[idx];if(shp[it.id]&&confirm("Remove all your drawings on this picture?")){delete shp[it.id];save();render()}};
$("sGt").onchange=render;$("sCand").onchange=render;document.querySelectorAll("input[name=mode]").forEach(r=>r.onchange=render);
const full=(it,p)=>[Math.round((it.cx+p[0]/it.scale)*10)/10,Math.round((it.cy+p[1]/it.scale)*10)/10];
function csvText(){const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
 return "candidate_id,source_image,decision,decided_at,n_outlines,yellow_outline_reshaped,file\n"+items.filter(x=>dec[x.id]).map(x=>[x.id,x.image,dec[x.id].d,dec[x.id].t,(shp[x.id]||[]).length,adj[x.id]?"yes":"no",fname(x)].map(q).join(",")).join("\n")+"\n"}
function jsonText(){return JSON.stringify({version:2,note:"all points are in full-image pixel coordinates (x right, y down). adjusted_outline = the yellow outline reshaped by the expert; outlines = new outlines drawn by the expert",items:items.filter(x=>dec[x.id]||(shp[x.id]||[]).length||adj[x.id]).map(x=>({candidate_id:x.id,source_image:x.image,decision:dec[x.id]?dec[x.id].d:null,adjusted_outline:adj[x.id]?adj[x.id].map(p=>full(x,p)):null,outlines:(shp[x.id]||[]).map(s=>s.map(p=>full(x,p)))}))},null,1)}
function download(name,text,type){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type:type||"text/csv"}));a.download=name;a.click()}
$("csv").onclick=()=>download("decisions.csv",csvText());$("json").onclick=()=>download("edits.json",jsonText(),"application/json");
async function saveFolders(){if(!window.showDirectoryPicker){alert("This browser cannot write folders. Use the two download buttons instead.");return}
 const root=await showDirectoryPicker({mode:"readwrite"});const ap=await root.getDirectoryHandle("approve",{create:true}),rj=await root.getDirectoryHandle("reject",{create:true});
 const out=document.createElement("canvas");out.width=512;out.height=512;const oc=out.getContext("2d");
 for(const it of items){const d=dec[it.id];if(!d)continue;const dir=d.d==="approve"?ap:rj;let fh=await dir.getFileHandle(fname(it),{create:true});let w=await fh.createWritable();await w.write(b64(it.base));await w.close();
  if((shp[it.id]||[]).length||adj[it.id]){await ensure(it);paint(oc,it,false);const blob=await new Promise(r=>out.toBlob(r,"image/png"));fh=await dir.getFileHandle(fname(it,"__with_my_outlines.png"),{create:true});w=await fh.createWritable();await w.write(blob);await w.close()}}
 for(const [n,t] of [["decisions.csv",csvText()],["edits.json",jsonText()]]){const fh=await root.getFileHandle(n,{create:true});const w=await fh.createWritable();await w.write(t);await w.close()}alert("Saved.")}
$("folders").onclick=()=>saveFolders().catch(e=>alert("Not saved: "+e.message));
$("reset").onclick=()=>{if(confirm("Delete ALL your answers, drawings and reshaped outlines in this file and start over?")){dec={};shp={};adj={};save();idx=0;render()}};
$("unlock").onclick=async()=>{$("msg").textContent="Opening...";try{const d=await decrypt($("pw").value);items=d.items;$("title").textContent=d.title;load();idx=firstUndecided();
 $("lock").hidden=true;$("app").hidden=false;await render();$("msg").textContent=""}catch(e){$("msg").textContent="Wrong password, or the file is damaged."}};
$("pw").addEventListener("keydown",e=>{if(e.key==="Enter")$("unlock").click()});
$("ok").onclick=()=>decide("approve");$("no").onclick=()=>decide("reject");
$("back").onclick=()=>{if(idx>0){idx--;render()}};$("skip").onclick=()=>{if(idx<items.length-1){idx++;render()}};
document.addEventListener("keydown",e=>{if($("app").hidden||e.target.tagName==="INPUT"||e.ctrlKey||e.metaKey)return;const k=e.key.toLowerCase();
 if(k==="a")decide("approve");else if(k==="r")decide("reject");else if(k==="u")undo();else if(k==="n"){const r=document.querySelectorAll("input[name=mode]");r[mode()==="nudge"?1:0].checked=true;render()}else if(k==="b"&&idx>0){idx--;render()}else if(k==="s"&&idx<items.length-1){idx++;render()}});
</script></body></html>
"""
html = (HTML.replace("__SALT__", base64.b64encode(salt).decode()).replace("__IV__", base64.b64encode(iv).decode())
        .replace("__ITER__", str(a.iterations)).replace("__CT__", base64.b64encode(ct).decode()))
Path(a.out).write_text(html, encoding="utf-8")
print(f"wrote {a.out}: {len(items)} candidates, {Path(a.out).stat().st_size/1e6:.1f} MB, PBKDF2 iterations {a.iterations}")

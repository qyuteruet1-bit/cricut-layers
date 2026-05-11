from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
import zipfile
import base64
import os

app = Flask(__name__)
CORS(app)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#4CAF50">
<title>3D Layers</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#f0f0f0;min-height:100vh;max-width:500px;margin:0 auto;padding:1rem}
.card{background:#fff;border-radius:16px;padding:20px;margin-bottom:15px;box-shadow:0 2px 12px rgba(0,0,0,.08)}
h1{font-size:1.6rem;text-align:center;margin-bottom:5px}
.sub{text-align:center;color:#666;font-size:.9rem;margin-bottom:15px}
.btn-row{display:flex;gap:10px;margin-bottom:12px}
.btn-row button{flex:1}
.btn{display:block;width:100%;padding:14px;background:#4CAF50;color:#fff;border:none;border-radius:12px;font-size:1rem;font-weight:700;cursor:pointer;margin:10px 0;text-align:center}
.btn-outline{padding:14px;background:#fff;color:#4CAF50;border:2px solid #4CAF50;border-radius:12px;font-size:1rem;font-weight:700;cursor:pointer;text-align:center}
.btn:disabled{opacity:.5}
input[type=range]{width:100%;margin:10px 0}
.preview-img{max-width:100%;border-radius:12px;display:none}
.downloads{margin-top:15px}
.downloads a{display:block;padding:12px 15px;margin:5px 0;border-radius:10px;text-decoration:none;font-weight:600;font-size:.95rem}
.downloads a.layer{background:#e8f5e9;color:#2e7d32}
.downloads a.inst{background:#e3f2fd;color:#1565c0}
.downloads a.zip{background:#fff3e0;color:#e65100}
.spinner-wrap{text-align:center;padding:20px;display:none}
.spinner{display:inline-block;width:40px;height:40px;border:4px solid #ccc;border-top-color:#4CAF50;border-radius:50%;animation:s .8s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
#status{text-align:center;margin:10px 0;font-weight:600;color:#555}
</style>
</head>
<body>
<div class="card">
<h1>📸 3D Layers</h1>
<p class="sub">Photo → Cricut SVGs</p>
<div class="btn-row">
<button class="btn-outline" onclick="document.getElementById('fileInput').click()">🖼️ Gallery</button>
<button class="btn-outline" onclick="document.getElementById('cameraInput').click()">📷 Camera</button>
</div>
<input type="file" id="fileInput" accept="image/*" style="display:none">
<input type="file" id="cameraInput" accept="image/*" capture="environment" style="display:none">
<label>Layers: <span id="layerNum">5</span></label>
<input type="range" id="layers" min="3" max="8" value="5" step="1">
<button class="btn" onclick="process()">🎨 Generate</button>
</div>
<div class="spinner-wrap" id="spinner"><div class="spinner"></div></div>
<div id="status"></div>
<img class="preview-img" id="preview">
<div class="downloads" id="downloads"></div>
<script>
var sf=null
document.getElementById('fileInput').onchange=function(e){if(e.target.files[0])sf=e.target.files[0]}
document.getElementById('cameraInput').onchange=function(e){if(e.target.files[0])sf=e.target.files[0]}
document.getElementById('layers').oninput=function(){document.getElementById('layerNum').textContent=this.value}
async function process(){
if(!sf){alert('Choose an image first');return}
const b=document.querySelector('.btn');b.disabled=true
document.getElementById('spinner').style.display='block'
document.getElementById('status').textContent='Processing...'
document.getElementById('downloads').innerHTML=''
document.getElementById('preview').style.display='none'
const fd=new FormData();fd.append('image',sf);fd.append('layers',document.getElementById('layers').value)
try{
const r=await fetch('/process',{method:'POST',body:fd})
if(!r.ok)throw new Error(await r.text())
const d=await r.json()
document.getElementById('status').textContent='Done! '+d.layers.length+' layers.'
if(d.preview){document.getElementById('preview').src='data:image/svg+xml;base64,'+d.preview;document.getElementById('preview').style.display='block'}
let h='';d.layers.forEach((l,i)=>{h+='<a class="layer" href="data:image/svg+xml;base64,'+l.data+'" download="'+l.name+'">📄 '+l.name+'</a>'})
if(d.inst)h+='<a class="inst" href="data:image/svg+xml;base64,'+d.inst+'" download="instructions.svg">📐 Instructions</a>'
if(d.zip)h+='<a class="zip" href="data:application/zip;base64,'+d.zip+'" download="layers.zip">📦 ZIP All</a>'
document.getElementById('downloads').innerHTML=h
}catch(e){document.getElementById('status').textContent='Error: '+e.message}
finally{b.disabled=false;document.getElementById('spinner').style.display='none'}
}
</script>
</body>
</html>'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/process', methods=['POST'])
def process():
    try:
        f = request.files['image']
        n = int(request.form.get('layers', 5))
        
        img = Image.open(f.stream).convert('RGB')
        w, h = img.size
        if max(w,h) > 600:
            s = 600 / max(w,h)
            img = img.resize((int(w*s), int(h*s)))
            w, h = img.size
        
        arr = np.array(img).reshape(-1, 3).astype(float)
        km = KMeans(n_clusters=n, random_state=0, n_init=10)
        lbs = km.fit_predict(arr)
        ctrs = km.cluster_centers_.astype(int)
        lbl_img = lbs.reshape(h, w)
        
        layers = []
        colors = []
        
        for i in range(n):
            mask = (lbl_img == i)
            color = tuple(ctrs[i])
            colors.append(color)
            hex_c = '#{:02x}{:02x}{:02x}'.format(*color)
            
            # Find all pixels of this color and create rectangles for contiguous regions
            # Simple approach: scan rows and create horizontal strips
            svg_parts = []
            for y in range(h):
                x_start = None
                for x in range(w):
                    if mask[y, x]:
                        if x_start is None:
                            x_start = x
                    else:
                        if x_start is not None:
                            if x - x_start > 2:  # ignore tiny strips
                                svg_parts.append(f'<rect x="{x_start}" y="{y}" width="{x-x_start}" height="1" fill="{hex_c}" stroke="none"/>')
                            x_start = None
                if x_start is not None and w - x_start > 2:
                    svg_parts.append(f'<rect x="{x_start}" y="{y}" width="{w-x_start}" height="1" fill="{hex_c}" stroke="none"/>')
            
            svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            svg += ''.join(svg_parts)
            svg += '</svg>'
            
            layers.append({
                'name': f'layer_{i+1}.svg',
                'data': base64.b64encode(svg.encode()).decode(),
                'count': len(svg_parts)
            })
        
        # Instructions
        areas = [np.sum(lbl_img == i) for i in range(n)]
        order = sorted(range(n), key=lambda i: areas[i], reverse=True)
        
        ox, oy = 20, -20
        iw = w + abs(ox)*(n-1) + 40
        ih = h + abs(oy)*(n-1) + 40
        
        inst = f'<svg xmlns="http://www.w3.org/2000/svg" width="{iw}" height="{ih}" viewBox="0 0 {iw} {ih}">'
        inst += f'<rect width="100%" height="100%" fill="white"/>'
        
        for idx, i in enumerate(order):
            mask = (lbl_img == i)
            color = colors[i]
            hex_c = '#{:02x}{:02x}{:02x}'.format(*color)
            dx, dy = ox*idx+20, oy*idx + abs(oy)*(n-1)+20
            
            for y in range(h):
                xs = None
                for x in range(w):
                    if mask[y,x]:
                        if xs is None: xs = x
                    else:
                        if xs is not None and x-xs > 2:
                            inst += f'<rect x="{xs+dx}" y="{y+dy}" width="{x-xs}" height="1" fill="{hex_c}" stroke="none" opacity="0.9"/>'
                        xs = None
                if xs is not None and w-xs > 2:
                    inst += f'<rect x="{xs+dx}" y="{y+dy}" width="{w-xs}" height="1" fill="{hex_c}" stroke="none" opacity="0.9"/>'
            
            inst += f'<text x="{10+dx}" y="{h+25+dy}" fill="black" font-size="16" font-family="Arial" font-weight="bold">Layer {idx+1}</text>'
        
        inst += '</svg>'
        inst_b64 = base64.b64encode(inst.encode()).decode()
        
        # ZIP
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, 'w', zipfile.ZIP_DEFLATED) as zf:
            for l in layers:
                zf.writestr(l['name'], base64.b64decode(l['data']).decode())
            zf.writestr('instructions.svg', inst)
        zip_b64 = base64.b64encode(zb.getvalue()).decode()
        
        return jsonify({
            'layers': layers,
            'inst': inst_b64,
            'preview': inst_b64,
            'zip': zip_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
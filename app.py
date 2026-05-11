from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite
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
<meta name="apple-mobile-web-app-status-bar-style" content="black">
<meta name="theme-color" content="#4CAF50">
<title>3D Layers</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#f0f0f0;min-height:100vh;max-width:500px;margin:0 auto;padding:1rem;-webkit-tap-highlight-color:transparent}
.card{background:#fff;border-radius:16px;padding:20px;margin-bottom:15px;box-shadow:0 2px 12px rgba(0,0,0,.08)}
h1{font-size:1.6rem;text-align:center;margin-bottom:5px}
.sub{text-align:center;color:#666;font-size:.9rem;margin-bottom:15px}
input[type=file]{width:100%;padding:12px;border:2px dashed #ccc;border-radius:12px;font-size:1rem;margin-bottom:12px;background:#fafafa}
label{font-weight:600;display:block;margin-bottom:5px}
input[type=range]{width:100%;margin:10px 0}
.row{display:flex;align-items:center;gap:10px}
.btn{display:block;width:100%;padding:14px;background:#4CAF50;color:#fff;border:none;border-radius:12px;font-size:1.1rem;font-weight:700;cursor:pointer;margin:10px 0;text-align:center;text-decoration:none}
.btn:disabled{opacity:.5}
.btn.secondary{background:#2196F3}
.preview-img{max-width:100%;border-radius:12px;border:1px solid #ddd;display:none}
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
<h1>📸 3D Ensemble</h1>
<p class="sub">Photo → Cricut Layers</p>

<input type="file" id="fileInput" accept="image/*" capture="environment">

<label>Layers: <span id="layerNum">5</span></label>
<input type="range" id="layers" min="3" max="8" value="5" step="1">

<div class="row">
<label style="font-weight:400;font-size:.9rem">
<input type="checkbox" id="foam" checked> Show foam tape
</label>
</div>

<button class="btn" onclick="process()">🎨 Generate</button>
</div>

<div class="spinner-wrap" id="spinner"><div class="spinner"></div></div>
<div id="status"></div>
<img class="preview-img" id="preview">

<div class="downloads" id="downloads"></div>

<script>
document.getElementById('layers').oninput=function(){
document.getElementById('layerNum').textContent=this.value
}

async function process(){
const f=document.getElementById('fileInput').files[0]
if(!f){alert('Choose an image first');return}

const btn=document.querySelector('.btn')
btn.disabled=true
document.getElementById('spinner').style.display='block'
document.getElementById('status').textContent='Uploading…'
document.getElementById('downloads').innerHTML=''
document.getElementById('preview').style.display='none'

const fd=new FormData()
fd.append('image',f)
fd.append('layers',document.getElementById('layers').value)
fd.append('foam',document.getElementById('foam').checked)

try{
document.getElementById('status').textContent='Processing… (10-30s)'
const r=await fetch('/process',{method:'POST',body:fd})
if(!r.ok)throw new Error(await r.text())
const d=await r.json()
document.getElementById('status').textContent='Done!'

if(d.preview){
const img=document.getElementById('preview')
img.src='data:image/svg+xml;base64,'+d.preview
img.style.display='block'
}

let html=''
d.layers.forEach((l,i)=>{
html+=`<a class="layer" href="data:image/svg+xml;base64,${l.data}" download="${l.name}">📄 ${l.name} (tap to save)</a>`
})
if(d.instructions) html+=`<a class="inst" href="data:image/svg+xml;base64,${d.instructions}" download="instructions.svg">📐 Instructions SVG</a>`
if(d.zip) html+=`<a class="zip" href="data:application/zip;base64,${d.zip}" download="layers.zip">📦 Download All (ZIP)</a>`
document.getElementById('downloads').innerHTML=html

}catch(e){
document.getElementById('status').textContent='Error: '+e.message
}finally{
btn.disabled=false
document.getElementById('spinner').style.display='none'
}
}
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files['image']
        n_layers = int(request.form.get('layers', 5))
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        max_dim = 1000
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=n_layers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        label_img = labels.reshape(h, w)
        
        layers = []
        masks_list = []
        colors_list = []
        
        for i in range(n_layers):
            mask = (label_img == i).astype(np.uint8) * 255
            masks_list.append(mask)
            colors_list.append(centers[i])
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            dwg = svgwrite.Drawing(size=(w, h))
            dwg.add(dwg.rect(insert=(0,0), size=(w,h), fill='white'))
            color_rgb = svgwrite.rgb(*centers[i])
            
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                if len(points) < 3:
                    continue
                path_data = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x,y in points) + " Z"
                dwg.add(dwg.path(d=path_data, fill=color_rgb, stroke='none'))
            
            svg_str = dwg.tostring()
            layers.append({
                'name': f'layer_{i+1}.svg',
                'data': base64.b64encode(svg_str.encode()).decode()
            })
        
        areas = [np.sum(m > 0) for m in masks_list]
        order = np.argsort(areas)[::-1]
        
        offset_x = 25
        offset_y = -25
        inst_w = w + abs(offset_x) * (n_layers - 1)
        inst_h = h + abs(offset_y) * (n_layers - 1)
        
        inst_svg = svgwrite.Drawing(size=(inst_w, inst_h))
        inst_svg.add(inst_svg.rect(insert=(0,0), size=('100%','100%'), fill='white'))
        
        for idx, i in enumerate(order):
            mask = masks_list[i]
            color = colors_list[i]
            dx = offset_x * idx
            dy = offset_y * idx
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color_rgb = svgwrite.rgb(*color)
            
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                points = [(float(pt[0][0])+dx, float(pt[0][1])+dy) for pt in approx]
                if len(points) < 3:
                    continue
                path_data = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x,y in points) + " Z"
                inst_svg.add(inst_svg.path(d=path_data, fill=color_rgb, stroke='black', stroke_width=0.5))
            
            inst_svg.add(inst_svg.text(f"Layer {idx+1}", insert=(10+dx, h+20+dy),
                         fill='black', font_size='18px', font_family='Arial', font_weight='bold'))
            
            reg_gap = 30
            for mx, my in [(reg_gap, reg_gap), (w-reg_gap, reg_gap), (reg_gap, h-reg_gap), (w-reg_gap, h-reg_gap)]:
                mx += dx
                my += dy
                inst_svg.add(inst_svg.line(start=(mx-6,my), end=(mx+6,my), stroke='black', stroke_width=1))
                inst_svg.add(inst_svg.line(start=(mx,my-6), end=(mx,my+6), stroke='black', stroke_width=1))
        
        inst_str = inst_svg.tostring()
        inst_b64 = base64.b64encode(inst_str.encode()).decode()
        
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for layer in layers:
                zf.writestr(layer['name'], base64.b64decode(layer['data']).decode())
            zf.writestr('instructions.svg', inst_str)
        zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()
        
        return jsonify({
            'layers': layers,
            'instructions': inst_b64,
            'preview': inst_b64,
            'zip': zip_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2
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
.spinner{display:inline-block;width:40px;height:40px;border:4px solid #ccc;border-top-color:#4CAF50;border-radius:50%;animation:s.8s linear infinite}
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

def preprocess_image(img, max_dim=1200):
    """Resize + denoise so contours are clean"""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Bilateral filter keeps edges sharp but removes noise
    np_img = np.array(img)
    denoised = cv2.bilateralFilter(np_img, 9, 75, 75)
    return Image.fromarray(denoised), img.size

def color_quantize(img, k):
    """Reduce colors first, then cluster. This fixes speckles."""
    np_img = np.array(img)
    pixels = np_img.reshape(-1, 3).astype(np.float32)

    # MiniBatchKMeans is faster and more stable on Render
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=1000, n_init=3)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    quantized = centers[labels].reshape(np_img.shape)
    return quantized, centers, labels.reshape(img.size[1], img.size[0])

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/process', methods=['POST'])
def process():
    try:
        f = request.files['image']
        n = int(request.form.get('layers', 5))

        img = Image.open(f.stream).convert('RGB')

        # 1. Preprocess
        img, (w, h) = preprocess_image(img, 1200)

        # 2. Quantize colors FIRST - this is the main fix
        quantized_img, centers, lbl_img = color_quantize(img, n)

        layers = []
        layer_data = []

        for i in range(n):
            mask = (lbl_img == i).astype(np.uint8) * 255

            # Clean mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

            # Find external contours only - no holes in holes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter tiny noise
            contours = [c for c in contours if cv2.contourArea(c) > 100]

            if not contours:
                continue

            path_d = ""
            for cnt in contours:
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) >= 3:
                    path_d += f"M {approx[0][0][0]} {approx[0][0][1]} "
                    for pt in approx[1:]:
                        path_d += f"L {pt[0][0]} {pt[0][1]} "
                    path_d += "Z "

            if not path_d:
                continue

            color = tuple(centers[i])
            hex_c = '#{:02x}{:02x}{:02x}'.format(*color)
            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]

            layer_data.append({'color': hex_c, 'path': path_d, 'brightness': brightness, 'idx': i})

            svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            svg += f'<path d="{path_d}" fill="{hex_c}" stroke="none" fill-rule="evenodd"/>'
            svg += '</svg>'

            layers.append({
                'name': f'layer_{i+1}.svg',
                'data': base64.b64encode(svg.encode()).decode()
            })

        # 3. Sort by brightness for correct 3D stack
        # Dark = bottom, Light = top
        layer_data.sort(key=lambda x: x['brightness'])

        # 4. Build instruction SVG
        ox, oy = 15, -15
        iw = w + abs(ox)*(len(layer_data)-1) + 40
        ih = h + abs(oy)*(len(layer_data)-1) + 40

        inst = f'<svg xmlns="http://www.w3.org/2000/svg" width="{iw}" height="{ih}" viewBox="0 0 {iw} {ih}">'
        inst += f'<rect width="100%" height="100%" fill="white"/>'

        for idx, layer in enumerate(layer_data):
            dx, dy = ox*idx+20, oy*idx+20
            inst += f'<g transform="translate({dx}, {dy})">'
            inst += f'<path d="{layer["path"]}" fill="{layer["color"]}" stroke="#333" stroke-width="0.5" fill-rule="evenodd"/>'
            inst += f'</g>'
            inst += f'<text x="{10+dx}" y="{h+25+dy}" fill="black" font-size="14" font-family="Arial" font-weight="bold">Layer {idx+1}</text>'

        inst += '</svg>'
        inst_b64 = base64.b64encode(inst.encode()).decode()

        # Zip
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, 'w', zipfile.ZIP_DEFLATED) as zf:
            for l in layers:
                zf.writestr(l['name'], base64.b64decode(l['data']).decode())
            zf.writestr('instructions.svg', inst)
        zip_b64 = base64.b64encode(zb.getvalue()).decode()

        # Sort download layers to match instruction order
        layers_sorted = [layers[d['idx']] for d in layer_data]

        return jsonify({
            'layers': layers_sorted,
            'inst': inst_b64,
            'preview': inst_b64,
            'zip': zip_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
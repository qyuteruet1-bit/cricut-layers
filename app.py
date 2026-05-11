from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image, ImageFilter
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
.thumb-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.thumb-item{width:70px;text-align:center}
.thumb-item div{width:70px;height:70px;border-radius:8px;border:1px solid #ddd}
.thumb-item span{font-size:.7rem;display:block;margin-top:3px}
</style>
</head>
<body>
<div class="card">
<h1>📸 3D Ensemble</h1>
<p class="sub">Photo → Cricut Layers</p>

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
<div id="thumbnails" class="thumb-row"></div>
<img class="preview-img" id="preview">
<div class="downloads" id="downloads"></div>

<script>
var selectedFile=null
document.getElementById('fileInput').onchange=function(e){if(e.target.files[0])selectedFile=e.target.files[0]}
document.getElementById('cameraInput').onchange=function(e){if(e.target.files[0])selectedFile=e.target.files[0]}
document.getElementById('layers').oninput=function(){document.getElementById('layerNum').textContent=this.value}
async function process(){
if(!selectedFile){alert('Choose an image from Gallery or Camera first');return}
const btn=document.querySelector('.btn')
btn.disabled=true
document.getElementById('spinner').style.display='block'
document.getElementById('status').textContent='Processing...'
document.getElementById('downloads').innerHTML=''
document.getElementById('preview').style.display='none'
document.getElementById('thumbnails').innerHTML=''
const fd=new FormData()
fd.append('image',selectedFile)
fd.append('layers',document.getElementById('layers').value)
try{
const r=await fetch('/process',{method:'POST',body:fd})
if(!r.ok)throw new Error(await r.text())
const d=await r.json()
document.getElementById('status').textContent='✅ Done! Tap links below to save.'
if(d.preview){
document.getElementById('preview').src='data:image/svg+xml;base64,'+d.preview
document.getElementById('preview').style.display='block'
}
if(d.thumbnails){
let th=''
d.thumbnails.forEach(t=>{
th+=`<div class="thumb-item"><div style="background:${t.color}"></div><span>${t.name}</span></div>`
})
document.getElementById('thumbnails').innerHTML=th
}
let html=''
d.layers.forEach((l,i)=>{html+=`<a class="layer" href="data:image/svg+xml;base64,${l.data}" download="${l.name}">📄 ${l.name} (tap to save)</a>`})
if(d.instructions)html+=`<a class="inst" href="data:image/svg+xml;base64,${d.instructions}" download="instructions.svg">📐 Instructions</a>`
if(d.zip)html+=`<a class="zip" href="data:application/zip;base64,${d.zip}" download="layers.zip">📦 Download All (ZIP)</a>`
document.getElementById('downloads').innerHTML=html
}catch(e){document.getElementById('status').textContent='❌ Error: '+e.message}
finally{btn.disabled=false;document.getElementById('spinner').style.display='none'}
}
</script>
</body>
</html>'''

def trace_contours(mask_img, tolerance=2.5):
    """Trace contours from a binary PIL image using border following."""
    w, h = mask_img.size
    pixels = mask_img.load()
    visited = set()
    all_contours = []
    
    # Border following directions (8-connected)
    dirs = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
    
    for y in range(h):
        for x in range(w):
            if pixels[x, y] > 128 and (x, y) not in visited:
                # Check if it's a boundary pixel
                is_boundary = False
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x+dx, y+dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h or pixels[nx, ny] <= 128:
                        is_boundary = True
                        break
                
                if not is_boundary:
                    continue
                
                contour = []
                cx, cy = x, y
                start_dir = 0
                steps = 0
                max_steps = w * h
                
                while steps < max_steps:
                    contour.append((cx, cy))
                    visited.add((cx, cy))
                    
                    found = False
                    for i in range(8):
                        nd = (start_dir + i) % 8
                        nx, ny = cx + dirs[nd][0], cy + dirs[nd][1]
                        if 0 <= nx < w and 0 <= ny < h and pixels[nx, ny] > 128:
                            cx, cy = nx, ny
                            start_dir = (nd + 4) % 8
                            found = True
                            break
                    
                    if not found or (cx == x and cy == y and steps > 1):
                        break
                    steps += 1
                
                if len(contour) >= 10:
                    contour = simplify(contour, tolerance)
                    if len(contour) >= 6:
                        all_contours.append(contour)
    
    return all_contours


def simplify(points, eps):
    """Ramer-Douglas-Peucker simplification."""
    if len(points) <= 2:
        return points
    
    def dist_to_line(pt, a, b):
        ax, ay = a; bx, by = b; px, py = pt
        dx, dy = bx-ax, by-ay
        mag_sq = dx*dx + dy*dy
        if mag_sq == 0:
            return ((px-ax)**2 + (py-ay)**2) ** 0.5
        u = ((px-ax)*dx + (py-ay)*dy) / mag_sq
        if u < 0:
            ix, iy = ax, ay
        elif u > 1:
            ix, iy = bx, by
        else:
            ix, iy = ax + u*dx, ay + u*dy
        return ((px-ix)**2 + (py-iy)**2) ** 0.5
    
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = dist_to_line(points[i], points[0], points[end])
        if d > dmax:
            dmax = d
            index = i
    
    if dmax > eps:
        left = simplify(points[:index+1], eps)
        right = simplify(points[index:], eps)
        return left[:-1] + right
    
    return [points[0], points[end]]


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files['image']
        n_layers = int(request.form.get('layers', 5))
        
        img = Image.open(file.stream).convert('RGB')
        
        # Resize if too large
        max_dim = 800
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        
        w, h = img.size
        
        # Slight blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Get pixel data
        pixels_list = list(img.getdata())
        pixels_arr = np.array(pixels_list, dtype=np.float32)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_layers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_arr)
        centers = kmeans.cluster_centers_.astype(int)
        label_img = labels.reshape(h, w)
        
        layers = []
        masks_list = []
        colors_list = []
        thumbnails = []
        
        for i in range(n_layers):
            mask_array = (label_img == i).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_array, mode='L')
            masks_list.append(mask_img)
            color = tuple(centers[i])
            colors_list.append(color)
            
            # Trace contours
            contours = trace_contours(mask_img, tolerance=2.5)
            
            # Build SVG
            dwg = svgwrite.Drawing(size=(w, h))
            
            color_rgb = svgwrite.rgb(*color)
            
            for contour in contours:
                if len(contour) >= 3:
                    path_data = "M " + " L ".join(f"{x},{y}" for x, y in contour) + " Z"
                    
            svg_str = dwg.tostring()
            layers.append({
                'name': f'layer_{i+1}.svg',
                'data': base64.b64encode(svg_str.encode()).decode()
            })
            
            # Thumbnail color
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            thumbnails.append({
                'name': f'Layer {i+1}',
                'color': hex_color
            })
        
        # Sort by area for instruction sheet
        areas = [np.sum(np.array(m) > 0) for m in masks_list]
        order = np.argsort(areas)[::-1]
        
        # Build instruction SVG
        ox, oy = 25, -25
        inst_w = w + abs(ox) * (n_layers - 1)
        inst_h = h + abs(oy) * (n_layers - 1)
        
        inst_svg = svgwrite.Drawing(size=(inst_w, inst_h))
        inst_svg.add(inst_svg.rect(insert=(0,0), size=('100%','100%'), fill='none'))
        
        for idx, i in enumerate(order):
            mask_img = masks_list[i]
            color = colors_list[i]
            dx, dy = ox*idx, oy*idx
            
            contours = trace_contours(mask_img, tolerance=2.5)
            color_rgb = svgwrite.rgb(*color)
            
            for contour in contours:
                if len(contour) >= 3:
                    pts = [(x+dx, y+dy) for x, y in contour]
                    path_data = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
                    inst_svg.add(inst_svg.path(d=path_data, fill=color_rgb, stroke='black', stroke_width=0.5))
            
            inst_svg.add(inst_svg.text(f"Layer {idx+1}", insert=(10+dx, h+20+dy),
                         fill='black', font_size='18px', font_family='Arial', font_weight='bold'))
            
            reg_gap = 30
            for mx, my in [(reg_gap, reg_gap), (w-reg_gap, reg_gap), (reg_gap, h-reg_gap), (w-reg_gap, h-reg_gap)]:
                inst_svg.add(inst_svg.line(start=(mx+dx-6, my+dy), end=(mx+dx+6, my+dy), stroke='black', stroke_width=1))
                inst_svg.add(inst_svg.line(start=(mx+dx, my+dy-6), end=(mx+dx, my+dy+6), stroke='black', stroke_width=1))
        
        inst_str = inst_svg.tostring()
        inst_b64 = base64.b64encode(inst_str.encode()).decode()
        
        # Create ZIP
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
            'zip': zip_b64,
            'thumbnails': thumbnails
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
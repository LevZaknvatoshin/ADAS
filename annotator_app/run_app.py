"""
ADAS Road Annotator Pro — Local Web Runner & Dataset Accumulator
Автономный локальный веб-сервер для разметки дорог, полос и накопления датасета.
"""
import os
import sys
import json
import base64
import socket
import re
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def sanitize_filename(name):
    # Remove invalid characters for Windows filenames
    clean = re.sub(r'[\\/*?:"<>|]', '_', name).strip()
    return clean if clean else "sample"

class AnnotatorServerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable aggressive caching for API
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/list_dataset':
            os.makedirs(DATASET_DIR, exist_ok=True)
            files = os.listdir(DATASET_DIR)
            json_files = [f for f in files if f.endswith('.json')]
            samples = []
            for jf in json_files:
                base = os.path.splitext(jf)[0]
                # Look for matching image
                img_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
                    candidate = base + ext
                    if candidate in files:
                        img_file = candidate
                        break
                samples.append({
                    'id': base,
                    'json': jf,
                    'image': img_file
                })
            
            response = json.dumps({'status': 'ok', 'count': len(samples), 'samples': samples}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return

        # Serve static files as normal
        super().do_GET()

    def do_POST(self):
        if self.path == '/api/save_sample':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                payload = json.loads(body)

                username = sanitize_filename(payload.get('username', 'anonymous'))
                raw_base_name = payload.get('base_name', 'road')
                raw_base_name = os.path.splitext(raw_base_name)[0]
                # Strip trailing username suffixes like _User, _Лев, etc.
                raw_base_name = re.sub(r'_(User|user|anonymous|' + re.escape(username) + r')$', '', raw_base_name, flags=re.IGNORECASE)
                base_name = sanitize_filename(raw_base_name)

                # Target filename strictly: <base_name>_<username>
                combined_name = f"{base_name}_{username}"
                os.makedirs(DATASET_DIR, exist_ok=True)

                # Save JSON
                json_data = payload.get('json_data', {})
                json_data['annotator_username'] = username
                
                # Determine image extension from data url or default to .jpg
                image_data_url = payload.get('image_data_url', '')
                ext = '.jpg'
                image_bytes = None

                if image_data_url.startswith('data:image/'):
                    header, b64data = image_data_url.split(',', 1)
                    if 'png' in header:
                        ext = '.png'
                    elif 'jpeg' in header or 'jpg' in header:
                        ext = '.jpg'
                    image_bytes = base64.b64decode(b64data)

                final_image_name = f"{combined_name}{ext}"
                final_json_name = f"{combined_name}.json"

                # Update filename inside JSON
                if 'image' in json_data and isinstance(json_data['image'], dict):
                    json_data['image']['file_name'] = final_image_name

                # Save image file
                img_path = os.path.join(DATASET_DIR, final_image_name)
                if image_bytes:
                    with open(img_path, 'wb') as f:
                        f.write(image_bytes)

                # Save JSON file
                json_path = os.path.join(DATASET_DIR, final_json_name)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                # Count total samples
                total_samples = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.json')])

                print(f"[+] Успешно сохранен кадр в датасет:")
                print(f"    - Фото: {final_image_name}")
                print(f"    - Разметка: {final_json_name}")
                print(f"    - Автор: {username}")
                print(f"    - Всего в датасете: {total_samples}")

                resp_data = {
                    'status': 'ok',
                    'message': 'Кадр успешно сохранен в датасет',
                    'image_file': final_image_name,
                    'json_file': final_json_name,
                    'username': username,
                    'total_samples': total_samples,
                    'dataset_dir': DATASET_DIR
                }
                resp_bytes = json.dumps(resp_data, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return

            except Exception as e:
                err_resp = json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)
                return

        self.send_response(404)
        self.end_headers()

def find_free_port(start_port=8080):
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
            port += 1
    return 8080

def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    os.makedirs(DATASET_DIR, exist_ok=True)
    port = find_free_port(8080)
    server_address = ('127.0.0.1', port)
    
    print("=" * 65)
    print("  [ADAS ROAD & LANE ANNOTATOR PRO]")
    print("  Интерактивная разметка полос и авто-накопление датасета")
    print("=" * 65)
    print(f"  [+] Сервер:        http://localhost:{port}")
    print(f"  [+] Папка датасета: {DATASET_DIR}")
    print("  [+] Открываем веб-интерфейс в браузере...")
    print("=" * 65)
    print("  (Для завершения нажмите Ctrl+C)")
    print()

    # Open browser
    webbrowser.open(f"http://localhost:{port}")

    httpd = HTTPServer(server_address, AnnotatorServerHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Сервер остановлен.")
        httpd.server_close()

if __name__ == '__main__':
    main()

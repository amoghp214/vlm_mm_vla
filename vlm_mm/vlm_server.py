import os
import sys
import json
import yaml
import argparse
from pathlib import Path

import http.server
import socketserver
import threading
import traceback
import urllib.parse

from vlm_mm import VLMMM

class _RequestHandler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, obj, code=200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/status":
                self._send_json({"status": "ok", "message": "VLMMM server running"})
            else:
                self._send_json({"error": "unknown endpoint"}, code=404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8') if length > 0 else ""
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}

            try:
                if parsed.path == "/run_step":
                    debug = bool(data.get("debug", False))
                    outputs = GLOBAL_VLMMM.step(debug=debug)
                    self._send_json({"status": "ok", "outputs": outputs})
                    return

                if parsed.path == "/analyse_performance":
                    # Use the server-side configured context video path from the VLMMM instance.
                    max_frames = int(data.get("max_frames", 10))
                    debug = bool(data.get("debug", False))
                    try:
                        analysis = GLOBAL_VLMMM.analyse_performance(max_frames=max_frames, debug=debug)
                        self._send_json({"status": "ok", "analysis": analysis})
                    except Exception as e:
                        self._send_json({"error": str(e), "type": type(e).__name__}, code=500)
                    return

                # unknown endpoint
                self._send_json({"error": "unknown endpoint"}, code=404)
            except Exception as e:
                # Print full traceback to server stderr for root-cause debugging
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                # If VLM_SERVER_DEBUG=1 include the traceback in the HTTP JSON response (dev only)
                if os.environ.get("VLM_SERVER_DEBUG", "0") == "1":
                    self._send_json({"error": str(e), "type": type(e).__name__, "traceback": tb}, code=500)
                else:
                    self._send_json({"error": str(e), "type": type(e).__name__}, code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLMMM server with a YAML config file.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="vlm_config.yaml",
        help="Path to YAML config file containing model_name, vlm_mm_context_dir, vlm_mm_prompts_dir, original_prompt, curr_image_path",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists(): 
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() not in (".yaml", ".yml"): 
        raise ValueError(f"Config file must be a YAML file with extension .yaml or .yml: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Read config keys (support a couple of alternative key names)
    model_name = config["vlmmm"].get("model_name", "gemma3")
    vlm_mm_context_dir = config["vlmmm"]["vlm_mm_context_dir"]
    vlm_mm_prompts_dir = config["vlmmm"]["vlm_mm_prompts_dir"]
    original_prompt = config["vlmmm"]["original_prompt"]
    curr_image_path = config["vlmmm"]["curr_image_path"]
    context_video_path = config["vlmmm"]["context_video_path"]
    vlm_device = config["vlmmm"].get("device")
    # Basic validation
    missing = []
    if original_prompt is None:
        missing.append("original_prompt (or vla_original_prompt)")
    if curr_image_path is None:
        missing.append("curr_image_path (or curr_image or vla_curr_image_path)")
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    # Instantiate VLMMM using config values
    vlm_mm = VLMMM(
        model_name=model_name,
        vlm_mm_context_dir=vlm_mm_context_dir,
        vlm_mm_prompts_dir=vlm_mm_prompts_dir,
        vla_original_prompt=original_prompt,
        vla_curr_image_path=curr_image_path,
        vla_context_video_path=context_video_path,
        vlm_device=vlm_device
    )

    # Expose the vlm_mm instance to the request handler via a global
    GLOBAL_VLMMM = vlm_mm

    # Server config
    HOST = os.environ.get("VLM_SERVER_HOST", "0.0.0.0")
    PORT = int(os.environ.get("VLM_SERVER_PORT", "8000"))

    with socketserver.ThreadingTCPServer((HOST, PORT), _RequestHandler) as httpd:
        print(f"VLMMM server listening on http://{HOST}:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down VLMMM server.")
            httpd.shutdown()
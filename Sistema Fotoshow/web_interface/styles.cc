/* Estilos modernos y limpios para una app de escritorio */
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background-color: #f0f2f5; color: #333; }
.app-container { display: flex; flex-direction: column; height: 100vh; }
.app-header { background-color: #ffffff; padding: 0 1.5rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex-shrink: 0; }
.logo { height: 32px; margin-right: 1rem; }
.app-header h1 { font-size: 1.5rem; color: #1a73e8; margin: 0; flex-grow: 1; }
.status-indicator { display: flex; align-items: center; font-size: 0.9rem; font-weight: 500; }
.status-dot { width: 10px; height: 10px; border-radius: 50%; background-color: #ccc; margin-right: 0.5rem; transition: background-color 0.3s; }
.status-dot.connected { background-color: #34a853; }
.status-dot.error { background-color: #ea4335; }

.app-main { flex-grow: 1; padding: 2rem; overflow-y: auto; }

.card { background: #ffffff; border-radius: 8px; padding: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
h2 { margin-top: 0; color: #202124; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.5rem; }
.form-group { margin-bottom: 1.5rem; }
.form-group label { display: block; margin-bottom: 0.5rem; font-weight: 500; }
.form-group input { width: 100%; padding: 0.75rem; border: 1px solid #dadce0; border-radius: 4px; font-size: 1rem; box-sizing: border-box; }
.form-group input:focus { outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.2); }
.folder-input-group { display: flex; gap: 0.5rem; }
.folder-input-group input { flex-grow: 1; }
.btn { padding: 0.75rem 1.5rem; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; font-weight: 500; transition: background-color 0.2s; }
.btn-primary { background-color: #1a73e8; color: white; }
.btn-primary:hover { background-color: #1558b8; }
.btn-secondary { background-color: #e8eaed; color: #3c4043; }
.btn-secondary:hover { background-color: #dadce0; }
.log-viewer { background-color: #2d2d2d; color: #0f0; font-family: 'Consolas', 'Monaco', monospace; padding: 1rem; height: 200px; overflow-y: scroll; border-radius: 4px; font-size: 0.85rem; }
.folder-list { list-style: none; padding: 0; }
.folder-list li { background: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; }
.folder-list li button { background: transparent; border: none; color: #d93025; cursor: pointer; font-size: 0.8rem; }

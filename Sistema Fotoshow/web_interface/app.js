document.addEventListener('DOMContentLoaded', () => {
    const mainContent = document.getElementById('main-content');
    const statusText = document.getElementById('status-text');
    const statusDot = document.querySelector('.status-dot');

    let currentConfig = {};

    async function fetchConfig() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                currentConfig = await response.json();
                return currentConfig;
            }
        } catch (error) {
            console.error("Error fetching config:", error);
        }
        return {};
    }

    async function saveConfig(config) {
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            return await response.json();
        } catch (error) {
            console.error("Error saving config:", error);
            return { success: false, error: error.message };
        }
    }

    function updateStatus(connected, text = 'Desconectado') {
        statusText.textContent = text;
        statusDot.className = `status-dot ${connected ? 'connected' : 'error'}`;
    }

    function renderSetupPage() {
        mainContent.innerHTML = `
            <div class="card">
                <h2>Configuración Inicial</h2>
                <p>Por favor, complete sus datos para conectar con el servicio FotoShow.</p>
                <form id="setup-form">
                    <div class="form-group">
                        <label for="name">Nombre Completo:</label>
                        <input type="text" id="name" required>
                    </div>
                    <div class="form-group">
                        <label for="email">Email:</label>
                        <input type="email" id="email" required>
                    </div>
                    <div class="form-group">
                        <label for="whatsapp">Número de WhatsApp:</label>
                        <input type="tel" id="whatsapp" placeholder="+5491112345678" required>
                    </div>
                    <div class="form-group">
                        <label for="folder">Carpeta Principal de Fotos:</label>
                        <div class="folder-input-group">
                            <input type="text" id="folder" readonly required>
                            <button type="button" class="btn btn-secondary" id="browse-btn">Examinar...</button>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary">Guardar y Conectar</button>
                </form>
            </div>
        `;

        document.getElementById('browse-btn').addEventListener('click', async () => {
            const folderPath = await window.pywebview.api.select_folder();
            if (folderPath) {
                document.getElementById('folder').value = folderPath;
            }
        });

        document.getElementById('setup-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            updateStatus(false, 'Registrando...');
            const newConfig = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                whatsapp_number: document.getElementById('whatsapp').value,
                monitored_folders: [document.getElementById('folder').value]
            };
            
            const result = await saveConfig(newConfig);
            if (result.success) {
                updateStatus(true, 'Conectado y Monitoreando');
                // Recargar la página para mostrar el dashboard
                setTimeout(() => window.location.reload(), 1500);
            } else {
                updateStatus(false, `Error: ${result.error}`);
                alert(`Error al guardar: ${result.error}`);
            }
        });
    }

    function renderDashboardPage(config) {
        mainContent.innerHTML = `
            <div class="card">
                <h2>Panel de Control</h2>
                <p><strong>Agente para:</strong> ${config.name || 'N/A'} (${config.email || 'N/A'})</p>
                
                <h3>Carpetas Monitoreadas</h3>
                <ul class="folder-list" id="folder-list">
                    ${config.monitored_folders.map(f => `<li>${f}</li>`).join('')}
                </ul>
                <button id="add-folder-btn" class="btn btn-secondary">Añadir Carpeta</button>
            </div>
            <div class="card">
                <h3>Log del Sistema</h3>
                <div class="log-viewer" id="log-viewer">Cargando log...</div>
            </div>
        `;

        document.getElementById('add-folder-btn').addEventListener('click', async () => {
            const folderPath = await window.pywebview.api.select_folder();
            if (folderPath && !config.monitored_folders.includes(folderPath)) {
                config.monitored_folders.push(folderPath);
                const result = await saveConfig(config);
                if (result.success) {
                    window.location.reload(); // Refrescar para mostrar la nueva carpeta
                }
            }
        });
        
        loadLogFile();
    }
    
    async function loadLogFile() {
        try {
            const logPath = await window.pywebview.api.get_log_file();
            const response = await fetch(`/api/read-log?path=${encodeURIComponent(logPath)}`);
            if (response.ok) {
                const logText = await response.text();
                document.getElementById('log-viewer').textContent = logText;
                document.getElementById('log-viewer').scrollTop = document.getElementById('log-viewer').scrollHeight;
            }
        } catch (error) {
            document.getElementById('log-viewer').textContent = "No se pudo cargar el archivo de log.";
        }
    }

    async function init() {
        const config = await fetchConfig();
        updateStatus(!!config.api_key, config.api_key ? 'Conectado' : 'Desconectado');

        if (config.api_key) {
            renderDashboardPage(config);
        } else {
            renderSetupPage();
        }
    }

    init();
});

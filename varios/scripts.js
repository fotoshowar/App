// ====== GLOBAL VARIABLES ======
let currentTab = 'dashboard';
let uploadedPhotos = [];
let searchResults = [];
let isUploading = false;
let isSearching = false;
let showWebcam = false;
let searchThreshold = 0.7;
let stats = { total_photos: 0, total_faces: 0 };
let isSearchMode = false;
let filteredPhotos = [];
let selectedPhoto = null;
let webcamStream = null;

// Variables para la carga múltiple de archivos
let selectedFiles = [];
let fileUploadQueue = [];
let currentUploadIndex = 0;
let totalUploadsCompleted = 0;
let totalUploadsFailed = 0;

// Backend URL configuration
function getBackendURL() {
    const hostname = window.location.hostname;
    const port = window.location.port;
    const protocol = window.location.protocol;
    
    if (hostname.includes('ngrok') || hostname.includes('ngrok-free.dev')) {
        return `${protocol}//${hostname}${port ? ':' + port : ''}`;
    }
    
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:8000';
    }
    
    return `${protocol}//${hostname}${port ? ':' + port : ''}`;
}

const BACKEND_URL = getBackendURL();
const API = `${BACKEND_URL}/api`;

// ====== INITIALIZATION ======
document.addEventListener('DOMContentLoaded', function() {
    console.log('Backend URL:', BACKEND_URL);
    console.log('API URL:', API);
    
    updateLastUpdateTime();
    document.getElementById('backend-url').textContent = BACKEND_URL;
    
    fetchPhotos();
    fetchStats();
    
    // Set up drag and drop for upload area
    setupDragAndDrop();
    
    // Update time every minute
    setInterval(updateLastUpdateTime, 60000);
});

// ====== NAVIGATION FUNCTIONS ======
function openSidebar() {
    document.getElementById('sidebar').classList.add('sidebar-open');
    document.getElementById('sidebar-overlay').classList.remove('hidden');
}

function closeSidebar() {
    document.getElementById('sidebar').classList.remove('sidebar-open');
    document.getElementById('sidebar-overlay').classList.add('hidden');
}

function switchTab(tabName) {
    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('nav-item-active');
        if (item.dataset.tab === tabName) {
            item.classList.add('nav-item-active');
        }
    });

    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Show selected tab content
    document.getElementById(tabName + '-tab').classList.add('active');

    // Update page title and subtitle
    const pageInfo = {
        dashboard: { icon: '📊', title: 'Dashboard', subtitle: 'Resumen general de tu colección' },
        upload: { icon: '📤', title: 'Subir Fotos', subtitle: 'Sube nuevas fotos para detectar caras' },
        search: { icon: '🔍', title: 'Buscar Cara', subtitle: 'Busca caras en tu colección' },
        gallery: { icon: '🖼️', title: 'Galería', subtitle: 'Explora todas tus fotos subidas' },
        results: { icon: '🎯', title: 'Resultados', subtitle: 'Resultados de tu última búsqueda' }
    };

    const info = pageInfo[tabName];
    document.getElementById('page-icon').textContent = info.icon;
    document.getElementById('page-title-text').textContent = info.title;
    document.getElementById('page-subtitle').textContent = info.subtitle;

    currentTab = tabName;
    closeSidebar();
}

// ====== UTILITY FUNCTIONS ======
function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('last-update-time').textContent = now.toLocaleTimeString();
}

function updateThreshold(value) {
    searchThreshold = parseFloat(value) / 100;
    document.getElementById('threshold-display').textContent = value + '%';
}

// ====== API FUNCTIONS ======
async function fetchPhotos() {
    try {
        console.log('Fetching photos from:', `${API}/photos`);
        const response = await fetch(`${API}/photos`, {
            headers: {
                'ngrok-skip-browser-warning': 'true'
            }
        });
        const data = await response.json();
        console.log('Photos response:', data);
        if (data.success) {
            uploadedPhotos = data.photos;
            renderPhotos();
        }
    } catch (error) {
        console.error('Error fetching photos:', error);
    }
}

async function fetchStats() {
    try {
        console.log('Fetching stats from:', `${BACKEND_URL}/api-status`);
        const response = await fetch(`${BACKEND_URL}/api-status`, {
            headers: {
                'ngrok-skip-browser-warning': 'true'
            }
        });
        const data = await response.json();
        console.log('Stats response:', data);
        if (data.database) {
            stats = data.database;
            updateStatsDisplay();
        }
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

function updateStatsDisplay() {
    document.getElementById('total-photos').textContent = stats.total_photos;
    document.getElementById('total-faces').textContent = stats.total_faces;
    document.getElementById('dashboard-total-photos').textContent = stats.total_photos;
    document.getElementById('dashboard-total-faces').textContent = stats.total_faces;
}

// ====== UPLOAD FUNCTIONS ======
function setupDragAndDrop() {
    const uploadArea = document.getElementById('upload-area');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-active');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-active');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-active');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            addFilesToQueue(files);
        }
    });
}

function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length > 0) {
        addFilesToQueue(files);
    }
}

function addFilesToQueue(files) {
    // Convert FileList to Array and add to selectedFiles
    const newFiles = Array.from(files);
    selectedFiles = [...selectedFiles, ...newFiles];
    
    // Show files list
    document.getElementById('files-list').classList.remove('hidden');
    
    // Update files count
    document.getElementById('files-count').textContent = `${selectedFiles.length} archivos`;
    
    // Render files list
    renderFilesList();
}

function renderFilesList() {
    const container = document.getElementById('files-container');
    container.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.id = `file-item-${index}`;
        
        // Determine file status
        let statusClass = 'file-status-pending';
        let statusText = 'Pendiente';
        
        if (file.status === 'uploading') {
            statusClass = 'file-status-uploading';
            statusText = 'Subiendo...';
        } else if (file.status === 'success') {
            statusClass = 'file-status-success';
            statusText = 'Completado';
        } else if (file.status === 'error') {
            statusClass = 'file-status-error';
            statusText = 'Error';
        }
        
        // Format file size
        const fileSize = formatFileSize(file.size);
        
        fileItem.innerHTML = `
            <div class="file-icon">📷</div>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${fileSize}</div>
                ${file.status === 'uploading' ? `
                    <div class="file-progress">
                        <div class="file-progress-bar" id="progress-${index}" style="width: ${file.progress || 0}%"></div>
                    </div>
                ` : ''}
            </div>
            <div class="file-status ${statusClass}">${statusText}</div>
        `;
        
        container.appendChild(fileItem);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function clearFileList() {
    selectedFiles = [];
    document.getElementById('files-list').classList.add('hidden');
    document.getElementById('file-input').value = '';
}

async function uploadAllFiles() {
    if (selectedFiles.length === 0 || isUploading) return;
    
    isUploading = true;
    currentUploadIndex = 0;
    totalUploadsCompleted = 0;
    totalUploadsFailed = 0;
    
    // Update UI
    document.getElementById('upload-prompt').classList.add('hidden');
    document.getElementById('upload-status').classList.remove('hidden');
    document.getElementById('upload-area').classList.add('uploading');
    
    // Initialize all files as pending
    selectedFiles.forEach(file => {
        file.status = 'pending';
        file.progress = 0;
    });
    
    renderFilesList();
    
    // Process files one by one
    await processUploadQueue();
    
    // Final summary
    isUploading = false;
    document.getElementById('upload-prompt').classList.remove('hidden');
    document.getElementById('upload-status').classList.add('hidden');
    document.getElementById('upload-area').classList.remove('uploading');
    
    // Show summary
    alert(`Proceso completado:\n${totalUploadsCompleted} archivos subidos con éxito\n${totalUploadsFailed} archivos con errores`);
    
    // Refresh data
    fetchPhotos();
    fetchStats();
    
    // Clear the list after successful upload
    if (totalUploadsCompleted > 0) {
        clearFileList();
    }
}

async function processUploadQueue() {
    while (currentUploadIndex < selectedFiles.length) {
        const file = selectedFiles[currentUploadIndex];
        
        // Update file status to uploading
        file.status = 'uploading';
        file.progress = 0;
        renderFilesList();
        
        try {
            await uploadSingleFile(file, currentUploadIndex);
            totalUploadsCompleted++;
        } catch (error) {
            console.error(`Error uploading ${file.name}:`, error);
            file.status = 'error';
            totalUploadsFailed++;
        }
        
        currentUploadIndex++;
    }
}

async function uploadSingleFile(file, index) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append('file', file);
        
        const xhr = new XMLHttpRequest();
        
        // Update progress
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                file.progress = percentComplete;
                
                // Update progress bar
                const progressBar = document.getElementById(`progress-${index}`);
                if (progressBar) {
                    progressBar.style.width = `${percentComplete}%`;
                }
            }
        });
        
        // Handle response
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        file.status = 'success';
                        file.facesDetected = response.faces_detected;
                        resolve(response);
                    } else {
                        file.status = 'error';
                        file.errorMessage = response.detail || 'Error desconocido';
                        reject(new Error(file.errorMessage));
                    }
                } catch (e) {
                    file.status = 'error';
                    file.errorMessage = 'Error parsing response';
                    reject(new Error(file.errorMessage));
                }
            } else {
                file.status = 'error';
                file.errorMessage = `HTTP ${xhr.status}`;
                reject(new Error(file.errorMessage));
            }
            
            renderFilesList();
        });
        
        // Handle errors
        xhr.addEventListener('error', () => {
            file.status = 'error';
            file.errorMessage = 'Network error';
            renderFilesList();
            reject(new Error(file.errorMessage));
        });
        
        // Send request
        xhr.open('POST', `${API}/upload-photo`, true);
        xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');
        xhr.send(formData);
    });
}

// ====== WEBCAM FUNCTIONS ======
async function toggleWebcam() {
    const webcamSection = document.getElementById('webcam-section');
    const webcamBtn = document.getElementById('webcam-btn');
    const webcamVideo = document.getElementById('webcam');
    const webcamPlaceholder = document.getElementById('webcam-placeholder');

    if (!showWebcam) {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user' } 
            });
            webcamVideo.srcObject = webcamStream;
            webcamVideo.classList.remove('hidden');
            webcamPlaceholder.classList.add('hidden');
            webcamSection.classList.remove('hidden');
            webcamBtn.textContent = '📷 Cerrar Cámara';
            showWebcam = true;
        } catch (error) {
            console.error('Error accessing webcam:', error);
            alert('Error accediendo a la cámara: ' + error.message);
        }
    } else {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
        webcamVideo.classList.add('hidden');
        webcamPlaceholder.classList.remove('hidden');
        webcamSection.classList.add('hidden');
        webcamBtn.textContent = '📷 Usar Cámara Web';
        showWebcam = false;
    }
}

async function capturePhoto() {
    if (isSearching || !webcamStream) return;

    const webcamVideo = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = webcamVideo.videoWidth;
    canvas.height = webcamVideo.videoHeight;
    ctx.drawImage(webcamVideo, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'webcam-capture.jpg');
        await performSearch(formData);
    }, 'image/jpeg', 0.8);
}

// ====== SEARCH FUNCTIONS ======
async function searchWithFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    await performSearch(formData);
    event.target.value = '';
}

async function performSearch(formData) {
    isSearching = true;
    const captureBtn = document.getElementById('capture-btn');
    if (captureBtn) {
        captureBtn.textContent = 'Buscando...';
        captureBtn.disabled = true;
    }

    try {
        console.log('Searching faces at:', API + '/search-face?threshold=' + searchThreshold);
        const response = await fetch(API + '/search-face?threshold=' + searchThreshold, {
            method: 'POST',
            headers: {
                'ngrok-skip-browser-warning': 'true'
            },
            body: formData
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);

        // Verificar el tipo de contenido de la respuesta
        const contentType = response.headers.get('content-type');
        console.log('Content-Type:', contentType);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response body:', errorText);
            throw new Error('HTTP ' + response.status + ': ' + (errorText || 'Error desconocido'));
        }

        // Verificar si la respuesta es JSON
        if (!contentType || !contentType.includes('application/json')) {
            const responseText = await response.text();
            console.error('Non-JSON response:', responseText.substring(0, 200));
            throw new Error('El servidor no devolvió JSON. Posible error 404 o problema en el endpoint.');
        }

        const data = await response.json();
        console.log('Search response:', data);

        if (data.success) {
            searchResults = data.matches || [];
            
            if (data.matches_found > 0) {
                const matchedPhotoIds = [...new Set(data.matches.map(match => match.photo_id))];
                filteredPhotos = uploadedPhotos.filter(photo => matchedPhotoIds.includes(photo.id));
                isSearchMode = true;
                switchTab('gallery');
                
                alert('Encontré ' + data.matches_found + ' coincidencias en ' + filteredPhotos.length + ' fotos!');
            } else {
                filteredPhotos = [];
                isSearchMode = false;
                alert('No se encontraron coincidencias con el umbral actual.');
            }
            
            renderPhotos();
            renderResults();
        } else {
            alert('Error en la búsqueda: ' + (data.detail || data.message || 'Error desconocido'));
        }
    } catch (error) {
        console.error('Search error:', error);
        if (error.message.includes('<!DOCTYPE')) {
            alert('Error: El servidor devolvió HTML en lugar de JSON. Verifica que el endpoint /api/search-face existe en tu backend.');
        } else {
            alert('Error buscando la cara: ' + error.message);
        }
    } finally {
        isSearching = false;
        if (captureBtn) {
            captureBtn.textContent = 'Capturar y Buscar';
            captureBtn.disabled = false;
        }
        if (showWebcam) {
            toggleWebcam(); // Close webcam after search
        }
    }
}

function clearSearchFilter() {
    isSearchMode = false;
    filteredPhotos = [];
    searchResults = [];
    renderPhotos();
}

// ====== PHOTO DELETION ======
async function deletePhoto(photoId) {
    if (!confirm('¿Estás seguro de que quieres eliminar esta foto?')) return;

    try {
        const response = await fetch(`${API}/photos/${photoId}`, {
            method: 'DELETE',
            headers: {
                'ngrok-skip-browser-warning': 'true'
            }
        });
        const data = await response.json();
        
        if (data.success) {
            alert('Foto eliminada correctamente');
            await fetchPhotos();
            await fetchStats();
        } else {
            alert('Error eliminando la foto: ' + (data.detail || 'Error desconocido'));
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Error eliminando la foto: ' + error.message);
    }
}

// ====== RENDERING FUNCTIONS ======
function renderPhotos() {
    const container = document.getElementById('photos-container');
    const photosToShow = isSearchMode ? filteredPhotos : uploadedPhotos;
    
    // Update gallery header
    const galleryTitle = document.getElementById('gallery-title');
    const gallerySubtitle = document.getElementById('gallery-subtitle');
    const searchModeIndicator = document.getElementById('search-mode-indicator');
    const clearSearchBtn = document.getElementById('clear-search-btn');

    if (isSearchMode) {
        galleryTitle.textContent = 'Fotos con Coincidencias';
        gallerySubtitle.textContent = filteredPhotos.length + ' fotos con coincidencias de búsqueda';
        searchModeIndicator.classList.remove('hidden');
        clearSearchBtn.classList.remove('hidden');
    } else {
        galleryTitle.textContent = 'Galería de Fotos';
        gallerySubtitle.textContent = uploadedPhotos.length + ' fotos en total';
        searchModeIndicator.classList.add('hidden');
        clearSearchBtn.classList.add('hidden');
    }

    if (photosToShow.length === 0) {
        const emptyTitle = isSearchMode ? 'No hay coincidencias' : 'No hay fotos aún';
        const emptySubtitle = isSearchMode 
            ? 'Intenta ajustar el umbral de búsqueda o busca otra persona'
            : '¡Sube algunas fotos para comenzar!';

        container.innerHTML = '<div class="empty-state">' +
            '<div class="empty-icon">🖼️</div>' +
            '<h3 class="empty-title">' + emptyTitle + '</h3>' +
            '<p class="empty-subtitle">' + emptySubtitle + '</p>' +
            (isSearchMode ? 
                '<div class="empty-actions">' +
                    '<button class="btn btn-primary" onclick="switchTab(\'search\')">' +
                        'Nueva Búsqueda' +
                    '</button>' +
                    '<button class="btn btn-ghost" onclick="clearSearchFilter()">' +
                        'Ver Todas las Fotos' +
                    '</button>' +
                '</div>' : '') +
            '</div>';
        return;
    }

    const photosGrid = document.createElement('div');
    photosGrid.className = 'photos-grid';

    photosToShow.forEach(photo => {
        let matchInfo = null;
        if (isSearchMode && searchResults.length > 0) {
            const photoMatches = searchResults.filter(match => match.photo_id === photo.id);
            if (photoMatches.length > 0) {
                const bestMatch = photoMatches.reduce((best, current) => 
                    current.similarity > best.similarity ? current : best
                );
                matchInfo = {
                    count: photoMatches.length,
                    bestSimilarity: bestMatch.similarity
                };
            }
        }

        const photoCard = document.createElement('div');
        photoCard.className = 'photo-card' + (isSearchMode ? ' photo-card-match' : '');

        const matchBadge = isSearchMode && matchInfo ? 
            '<div class="match-badge">🎯 ' + Math.round(matchInfo.bestSimilarity * 100) + '%</div>' : '';
        
        const matchInfoText = isSearchMode && matchInfo ? 
            '<p class="match-info">' + matchInfo.count + ' coincidencia' + (matchInfo.count > 1 ? 's' : '') + '</p>' : '';

        photoCard.innerHTML = '<div class="photo-image">' +
                '<img src="' + API + '/image/photo/' + photo.id + '" alt="' + photo.filename + '" ' +
                     'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">' +
                '<div class="photo-placeholder" style="display: none;">🖼️</div>' +
                '<div class="photo-badge">' + photo.faces_count + ' caras</div>' +
                matchBadge +
            '</div>' +
            '<div class="photo-info">' +
                '<h4 class="photo-title">' + photo.filename + '</h4>' +
                '<p class="photo-date">' + new Date(photo.upload_date).toLocaleDateString() + '</p>' +
                matchInfoText +
            '</div>';

        photosGrid.appendChild(photoCard);
    });

    container.innerHTML = '';
    container.appendChild(photosGrid);
}

function renderResults() {
    const container = document.getElementById('results-container');

    if (searchResults.length === 0) {
        container.innerHTML = '<div class="empty-state">' +
                '<div class="empty-icon">🎯</div>' +
                '<h3 class="empty-title">No hay resultados aún</h3>' +
                '<p class="empty-subtitle">Usa la función de búsqueda para encontrar coincidencias</p>' +
                '<button class="btn btn-primary" onclick="switchTab(\'search\')">' +
                    'Comenzar Búsqueda' +
                '</button>' +
            '</div>';
        return;
    }

    let resultsHTML = '<div class="results-section">' +
        '<div class="results-header">' +
            '<div class="results-info">' +
                '<h3 class="results-title">Resultados de Búsqueda</h3>' +
                '<p class="results-subtitle">' + searchResults.length + ' coincidencias encontradas</p>' +
            '</div>' +
            '<button class="btn btn-ghost" onclick="clearResults()">' +
                'Limpiar resultados' +
            '</button>' +
        '</div>' +
        '<div class="results-grid">';

    searchResults.forEach((result, index) => {
        resultsHTML += '<div class="result-card">' +
                '<div class="result-image">' +
                    '<img src="' + API + '/image/photo/' + result.photo_id + '" alt="' + result.photo_filename + '"' +
                         ' onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">' +
                    '<div class="result-placeholder" style="display: none;">🖼️</div>' +
                    '<div class="result-rank">#' + (index + 1) + '</div>' +
                    '<div class="result-similarity">' + Math.round(result.similarity * 100) + '%</div>' +
                '</div>' +
                '<div class="result-info">' +
                    '<h4 class="result-title">' + result.photo_filename + '</h4>' +
                    '<p class="result-score">Similitud: ' + Math.round(result.similarity * 100) + '%</p>' +
                    '<p class="result-id">ID: ' + result.face_id.slice(0, 8) + '...</p>' +
                '</div>' +
            '</div>';
    });

    resultsHTML += '</div></div>';
    container.innerHTML = resultsHTML;
}

function clearResults() {
    searchResults = [];
    renderResults();
}

// ====== PHOTO POPUP FUNCTIONS ======
function openPhotoPopup(photoId) {
    const photo = uploadedPhotos.find(p => p.id === photoId);
    if (!photo) return;

    selectedPhoto = photo;
    
    document.getElementById('popup-filename').textContent = photo.filename;
    document.getElementById('popup-date').textContent = new Date(photo.upload_date).toLocaleDateString('es-ES', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    document.getElementById('popup-faces-count').textContent = photo.faces_count;
    document.getElementById('popup-file-size').textContent = photo.file_size ? 
        Math.round(photo.file_size / 1024) + ' KB' : 'N/A';
    document.getElementById('popup-photo-id').textContent = photo.id;

    const popupImage = document.getElementById('popup-image');
    const popupPlaceholder = document.getElementById('popup-placeholder');
    
    popupImage.src = API + '/image/photo/' + photo.id;
    popupImage.style.display = 'block';
    popupPlaceholder.classList.add('hidden');
    
    popupImage.onerror = function() {
        this.style.display = 'none';
        popupPlaceholder.classList.remove('hidden');
    };

    const description = 'Esta foto contiene ' + photo.faces_count + ' ' + (photo.faces_count === 1 ? 'cara detectada' : 'caras detectadas') + '. Puedes imprimir esta imagen en formato A5 optimizado para una mejor calidad de impresión.';
    document.getElementById('popup-description').textContent = description;

    document.getElementById('photo-popup-overlay').classList.remove('hidden');
}

function openPhotoPopupFromResult(photoId) {
    openPhotoPopup(photoId);
}

function closePhotoPopup() {
    document.getElementById('photo-popup-overlay').classList.add('hidden');
    selectedPhoto = null;
}

function printPhoto() {
    if (!selectedPhoto) return;

    const printWindow = window.open('', '_blank');
    const imageUrl = API + '/image/photo/' + selectedPhoto.id;
    
    printWindow.document.write('<!DOCTYPE html>' +
        '<html>' +
            '<head>' +
                '<title>Imprimir Foto - ' + selectedPhoto.filename + '</title>' +
                '<style>' +
                    '@page {' +
                        'size: A5;' +
                        'margin: 0;' +
                    '}' +
                    'body {' +
                        'margin: 0;' +
                        'padding: 0;' +
                        'display: flex;' +
                        'justify-content: center;' +
                        'align-items: center;' +
                        'height: 100vh;' +
                        'background: white;' +
                    '}' +
                    '.print-container {' +
                        'width: 148mm;' +
                        'height: 210mm;' +
                        'display: flex;' +
                        'justify-content: center;' +
                        'align-items: center;' +
                        'padding: 10mm;' +
                        'box-sizing: border-box;' +
                    '}' +
                    '.print-image {' +
                        'max-width: 100%;' +
                        'max-height: 100%;' +
                        'object-fit: contain;' +
                        'box-shadow: 0 2px 10px rgba(0,0,0,0.1);' +
                    '}' +
                    '@media print {' +
                        'body {' +
                            'print-color-adjust: exact;' +
                            '-webkit-print-color-adjust: exact;' +
                        '}' +
                    '}' +
                '</style>' +
            '</head>' +
            '<body>' +
                '<div class="print-container">' +
                    '<img src="' + imageUrl + '" alt="' + selectedPhoto.filename + '" class="print-image" onload="window.focus(); window.print(); window.close();" />' +
                '</div>' +
            '</body>' +
        '</html>');
    
    printWindow.document.close();
}

// ====== CLEANUP ======
window.addEventListener('beforeunload', function() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
});

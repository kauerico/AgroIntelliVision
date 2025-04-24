document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const resultContainer = document.getElementById('resultContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

    // Drag and Drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadArea.classList.add('active');
    }

    function unhighlight() {
        uploadArea.classList.remove('active');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFiles(files);
        }
    }

    // File selection
    uploadBtn.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function(event) {
        if (event.target.files.length) {
            handleFiles(event.target.files);
        }
    });

    function handleFiles(files) {
        const file = files[0];
        if (file) {
            // Check file type
            if (!file.type.match('image.*')) {
                showError('Por favor, selecione um arquivo de imagem (JPG, PNG)');
                return;
            }
            
            // Check file size (5MB max)
            if (file.size > 5 * 1024 * 1024) {
                showError('O arquivo é muito grande (máximo 5MB)');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('d-none');
                resultContainer.innerHTML = `
                    <div class="text-center py-4">
                        <i class="bi bi-card-image" style="font-size: 3rem; color: #ccc;"></i>
                        <h4 class="mt-3">Imagem pronta para análise</h4>
                        <p class="text-muted">Clique em "Analisar Imagem" para obter o diagnóstico</p>
                    </div>
                `;
                scrollToPreview();
            };
            reader.readAsDataURL(file);
        }
    }

    function scrollToPreview() {
        previewContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function showError(message) {
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle-fill"></i> ${message}
            </div>
        `;
    }

    // Analyze button
    analyzeBtn.addEventListener('click', function() {
        if (!fileInput.files.length) {
            showError('Por favor, selecione uma imagem primeiro');
            return;
        }
        
        loadingModal.show();
        
        // Simulate API call
        setTimeout(function() {
            loadingModal.hide();
            showResults();
        }, 3000);
    });

    function showResults() {
        // In a real app, this would come from your API response
        const mockResults = {
            disease: 'Mosaico',
            confidence: 92.7,
            description: 'O mosaico é causado por vírus e caracterizado por manchas claras e escuras nas folhas.',
            treatment: 'Recomenda-se a eliminação de plantas infectadas e controle de vetores (pulgões).'
        };
        
        resultContainer.innerHTML = `
            <div class="result-card">
                <div class="card-header">
                    <i class="bi bi-check-circle-fill me-2"></i> Diagnóstico Completo
                </div>
                <div class="card-body">
                    <h3 class="disease-name">${mockResults.disease}</h3>
                    <p>${mockResults.description}</p>
                    
                    <div class="mt-4">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Confiança do diagnóstico:</span>
                            <span class="confidence-value">${mockResults.confidence}%</span>
                        </div>
                        <div class="confidence-meter">
                            <div class="confidence-level" style="width: ${mockResults.confidence}%"></div>
                        </div>
                    </div>
                    
                    <div class="treatment-info">
                        <h5 class="treatment-title"><i class="bi bi-heart-pulse-fill me-2"></i> Recomendações de Tratamento</h5>
                        <p>${mockResults.treatment}</p>
                    </div>
                </div>
            </div>
        `;
        
        // Scroll to results
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }

    // Reset button
    resetBtn.addEventListener('click', function() {
        fileInput.value = '';
        previewContainer.classList.add('d-none');
        resultContainer.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-card-image" style="font-size: 3rem; color: #ccc;"></i>
                <h4 class="mt-3">Nenhum resultado disponível</h4>
                <p class="text-muted">Envie uma imagem e clique em "Analisar" para obter o diagnóstico</p>
            </div>
        `;
    });
});
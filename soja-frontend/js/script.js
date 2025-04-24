document.addEventListener('DOMContentLoaded', function() {
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const resultContainer = document.getElementById('resultContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

    uploadBtn.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
        }
    });

    analyzeBtn.addEventListener('click', function() {
        loadingModal.show();
        setTimeout(function() {
            loadingModal.hide();
            resultContainer.innerHTML = `
                <div class="card">
                    <div class="card-header bg-success text-white">
                        Diagnóstico Completo
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">Doença Detectada: Mosaico</h5>
                        <p class="card-text">Confiança: 92.7%</p>
                    </div>
                </div>
            `;
        }, 2000);
    });

    resetBtn.addEventListener('click', function() {
        previewContainer.classList.add('d-none');
        resultContainer.innerHTML = '<p class="text-muted my-5">Clique em "Analisar" para obter o diagnóstico</p>';
        fileInput.value = '';
    });
});

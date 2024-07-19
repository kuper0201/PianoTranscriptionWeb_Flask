document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('uploadForm');

    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file && file.type.match('audio.*')) {
            submitBtn.disabled = false;
        } else {
            submitBtn.disabled = true;
            alert('Please select an audio file.');
            this.value = '';
        }
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!fileInput.files[0]) {
            alert('Please select a file before submitting.');
            return;
        }

        loadingOverlay.style.display = 'flex';
        downloadContainer.style.display = 'none';

        const formData = new FormData(this);

        fetch('/fileUpload', {method: 'POST', body: formData})
        .then(response => response.json())
        .then(data => {
            loadingOverlay.style.display = 'none';
            if (data.success) {
                alert(data.message);
                downloadLink.href = data.downloadUrl;
                downloadContainer.style.display = 'block';
                form.reset();
                submitBtn.disabled = true;
                submitBtn.classList.add('disabled');
            } else {
                alert('Upload failed: ' + data.message);
            }
        })
        .catch(error => {
            loadingOverlay.style.display = 'none';
            alert('An error occurred during upload.');
            console.error('Error:', error);
        });
    });
});
function toggleInput() {
    const scanType = document.getElementById('scan_type').value;
    const imageInput = document.getElementById('imageInput');
    const metadataInput = document.getElementById('metadataInput');

    if (scanType === 'metadata') {
        imageInput.style.display = 'none';
        metadataInput.style.display = 'block';
    } else {
        imageInput.style.display = 'block';
        metadataInput.style.display = 'none';
    }
}

// Run on page load to set initial state
document.addEventListener('DOMContentLoaded', toggleInput);
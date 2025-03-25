document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const uploadContainer = document.getElementById('upload-container');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const fileInput = document.getElementById('file-input');
    const changeImageBtn = document.getElementById('change-image');
    const submitImageBtn = document.getElementById('submit-image');
    const predictionForm = document.getElementById('prediction-form');
    const formFileInput = document.getElementById('form-file-input');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadArea.classList.add('highlight');
    }

    function unhighlight() {
        uploadArea.classList.remove('highlight');
    }

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files);
        }
    }

    // Handle file input change
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFiles(this.files);
        }
    });

    // Process the files
    // Function to handle file preview display
    function handleFiles(files) {
        const file = files[0]; // Only handle the first file if multiple are selected
        
        // Validate the file is an image
        if (!file.type.match('image.*')) {
            alert('Please select an image file (JPEG, PNG, GIF, etc.)');
            return;
        }
        
        // Check file size (limit to 5MB)
        if (file.size > 5 * 1024 * 1024) {
            alert('File is too large. Please select an image less than 5MB.');
            return;
        }
        
        // Display preview with preloading image handling
        const reader = new FileReader();
        
        // Show loading state
        previewContainer.style.display = 'flex';
        uploadArea.style.display = 'none';
        previewImage.src = 'static/images/loading.gif'; // Optional: use a loading spinner
        
        reader.onload = function(e) {
            // Create a temporary image to get dimensions
            const img = new Image();
            img.onload = function() {
                // Determine if image is landscape or portrait
                if (img.width > img.height) {
                    previewImage.classList.add('landscape');
                    previewImage.classList.remove('portrait');
                } else {
                    previewImage.classList.add('portrait');
                    previewImage.classList.remove('landscape');
                }
                
                // Set the actual image
                previewImage.src = e.target.result;
            };
            
            img.src = e.target.result;
        };
        
        reader.readAsDataURL(file);
        
        // Store the file for submission
        storeFile(file);
    }

    // Store the file in the form
    function storeFile(file) {
        // Create a new FileList object with just our file
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        formFileInput.files = dataTransfer.files;
    }

    // Handle change image button
    changeImageBtn.addEventListener('click', function() {
        uploadArea.style.display = 'flex';
        previewContainer.style.display = 'none';
        fileInput.value = '';
        formFileInput.value = '';
    });

    // Handle submit button
    submitImageBtn.addEventListener('click', function() {
        if (!formFileInput.files.length) {
            alert('Please select an image first.');
            return;
        }
        
        // Submit the form
        predictionForm.submit();
    });
});
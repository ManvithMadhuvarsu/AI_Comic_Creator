document.addEventListener('DOMContentLoaded', function() {
    const storyForm = document.getElementById('storyForm');
    const formatType = document.getElementById('formatType');
    const progressArea = document.getElementById('progressArea');
    const resultArea = document.getElementById('resultArea');
    const downloadLink = document.getElementById('downloadLink');
    const getRandomStoryBtn = document.getElementById('getRandomStory');

    getRandomStoryBtn.addEventListener('click', async function() {
        const response = await fetch(`/get_random_story?format_type=${formatType.value}`);
        const data = await response.json();
        document.querySelector('textarea[name="story_text"]').value = data.story;
    });

    storyForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(storyForm);
        const files = formData.getAll('images');
        
        if (files.length < 4 || files.length > 5) {
            alert('Please upload 4-5 images');
            return;
        }

        // Show progress
        progressArea.classList.remove('d-none');
        resultArea.classList.add('d-none');
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            if (data.success) {
                downloadLink.href = data.pdf_url;
                resultArea.classList.remove('d-none');
            }
        } catch (error) {
            alert('An error occurred during processing');
        } finally {
            progressArea.classList.add('d-none');
        }
    });
});
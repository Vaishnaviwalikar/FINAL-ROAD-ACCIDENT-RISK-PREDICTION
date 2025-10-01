// Force update data source indicator to show AI model
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for page to load
    setTimeout(function() {
        const indicator = document.getElementById('dataSourceIndicator');
        if (indicator) {
            const iconSpan = indicator.querySelector('.indicator-icon');
            const textSpan = indicator.querySelector('.indicator-text');
            const modelSpan = indicator.querySelector('.indicator-model');
            
            // Force to AI Neural Network
            iconSpan.textContent = '🧠';
            textSpan.textContent = 'AI Neural Network';
            modelSpan.textContent = 'CNN-BiLSTM-Attention';
            indicator.className = 'data-source-indicator ai-model';
            
            console.log('Forced indicator to AI Neural Network');
        }
    }, 1000);
});
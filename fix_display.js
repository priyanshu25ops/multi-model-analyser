// Fixed displayResults function - paste this into browser console if needed

function displayResults(data) {
    const results = document.getElementById('results');
    results.style.display = 'block';
    const badge = document.getElementById('topPrediction');
    const barsContainer = document.getElementById('bars');
    barsContainer.innerHTML = '';

    // Handle Medical Report (has 'analysis' field)
    if (data.analysis) {
        badge.innerText = 'AI Medical Analysis';
        const analysisHtml = `
            <div class="bar-container" style="margin-bottom:1rem; padding:1.5rem; background:rgba(255,255,255,0.05); border-radius:8px;">
                <div style="color:#e0e0e0; line-height:1.8; white-space:pre-wrap;">${data.analysis}</div>
            </div>
        `;
        barsContainer.innerHTML = analysisHtml;
    } else {
        // Handle X-Ray/MRI (has 'probabilities' field)
        badge.innerText = `${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`;

        const sorted = Object.entries(data.probabilities).sort(([, a], [, b]) => b - a);

        sorted.forEach(([label, prob]) => {
            const percent = (prob * 100).toFixed(1);
            const html = `
            <div class="bar-container">
                <div class="bar-label">
                    <span>${label}</span>
                    <span>${percent}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
            barsContainer.innerHTML += html;
        });
    }

    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth' });
}

console.log('Fixed displayResults function loaded!');

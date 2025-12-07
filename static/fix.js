<script>
// Override displayResults function
    window.displayResults = function(data) {
    const results = document.getElementById('results');
    results.style.display = 'block';
    const badge = document.getElementById('topPrediction');
    const barsContainer = document.getElementById('bars');
    barsContainer.innerHTML = '';

    if (data.analysis) {
        // Medical Report response
        badge.innerText = 'AI Medical Analysis';
    barsContainer.innerHTML = `<div class="bar-container" style="padding:1.5rem;background:rgba(255,255,255,0.05);border-radius:8px;"><div style="color:#e0e0e0;line-height:1.8;white-space:pre-wrap;">${data.analysis}</div></div>`;
    } else {
        // X-Ray/MRI response
        badge.innerText = `${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`;
        const sorted = Object.entries(data.probabilities).sort(([, a], [, b]) => b - a);
        sorted.forEach(([label, prob]) => {
            const percent = (prob * 100).toFixed(1);
    barsContainer.innerHTML += `<div class="bar-container"><div class="bar-label"><span>${label}</span><span>${percent}%</span></div><div class="bar-bg"><div class="bar-fill" style="width:${percent}%"></div></div></div>`;
        });
    }
    results.scrollIntoView({behavior: 'smooth' });
};
    console.log('✅ Fixed displayResults loaded!');
</script>

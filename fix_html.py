#!/usr/bin/env python3
# Fix index.html displayResults function

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the displayResults function
old_line = 'badge.innerText = `${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`;'

new_code = '''if (data.analysis) {
                badge.innerText = "AI Medical Analysis";
                barsContainer.innerHTML = '<div style="padding:1.5rem;background:rgba(255,255,255,0.05);border-radius:8px;color:#e0e0e0;line-height:1.8;white-space:pre-wrap;">' + data.analysis + '</div>';
                results.scrollIntoView({ behavior: 'smooth' });
                return;
            }
            badge.innerText = `${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`;'''

content = content.replace(old_line, new_code)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed displayResults function!")

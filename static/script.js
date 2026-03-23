// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const clearBtn = document.getElementById('clearBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const shareBtn = document.getElementById('shareBtn');
const aboutBtn = document.getElementById('aboutBtn');
const contactBtn = document.getElementById('contactBtn');
const aboutModal = document.getElementById('aboutModal');
const closeModal = document.querySelector('.close');

// Supported plants list
const supportedPlants = [
    'Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 
    'Pepper', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Tomato'
];

// Plant icons mapping
const plantIcons = {
    'Apple': '🍎', 'Blueberry': '🫐', 'Cherry': '🍒', 'Corn': '🌽',
    'Grape': '🍇', 'Orange': '🍊', 'Peach': '🍑', 'Pepper': '🫑',
    'Potato': '🥔', 'Raspberry': '🍓', 'Soybean': '🌱', 'Squash': '🎃',
    'Strawberry': '🍓', 'Tomato': '🍅'
};

// Initialize the plants grid
function initPlantsGrid() {
    const plantsGrid = document.getElementById('plantsGrid');
    plantsGrid.innerHTML = supportedPlants.map(plant => `
        <div class="plant-item">
            <i>${plantIcons[plant] || '🌿'}</i>
            <span>${plant}</span>
        </div>
    `).join('');
}

// Event Listeners
if (uploadBtn) {
    uploadBtn.addEventListener('click', () => fileInput.click());
}

if (uploadArea) {
    uploadArea.addEventListener('click', () => fileInput.click());
}

fileInput.addEventListener('change', handleFileSelect);
clearBtn?.addEventListener('click', resetApp);
newAnalysisBtn?.addEventListener('click', resetApp);
shareBtn?.addEventListener('click', shareResults);
aboutBtn?.addEventListener('click', () => aboutModal.style.display = 'block');
contactBtn?.addEventListener('click', () => alert('📧 Contact us: support@plantguard.com'));
closeModal?.addEventListener('click', () => aboutModal.style.display = 'none');

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === aboutModal) {
        aboutModal.style.display = 'none';
    }
});

// Drag and drop functionality
uploadArea?.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#ff9800';
    uploadArea.style.background = 'rgba(76, 175, 80, 0.05)';
});

uploadArea?.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#4caf50';
    uploadArea.style.background = 'transparent';
});

uploadArea?.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#4caf50';
    uploadArea.style.background = 'transparent';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, or WebP)');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        
        // Analyze the image
        analyzeImage(file);
    };
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    // Show loading animation
    loadingSection.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please check if the server is running.');
    } finally {
        loadingSection.style.display = 'none';
    }
}

function displayResults(data) {
    const top = data.top_prediction;
    const alternatives = data.predictions.slice(1);
    
    // Update confidence badge
    const badge = document.getElementById('confidenceBadge');
    const confidenceLevel = top.confidence > 80 ? 'high' : (top.confidence > 60 ? 'medium' : 'low');
    badge.className = `confidence-badge ${confidenceLevel}`;
    badge.innerHTML = `${top.confidence}% Confidence`;
    
    // Display main result
    const mainResult = document.getElementById('mainResult');
    mainResult.innerHTML = `
        <div class="main-result">
            <i class="fas fa-leaf" style="font-size: 48px; color: #4caf50;"></i>
            <div class="plant-name">${top.plant}</div>
            <div class="disease-name">${top.disease}</div>
            <div class="confidence">${top.confidence}%</div>
            <p style="color: #5d6e5e;">Detection Confidence</p>
        </div>
    `;
    
    // Display alternatives
    if (alternatives.length > 0) {
        const alternativesSection = document.getElementById('alternativesSection');
        const alternativesList = document.getElementById('alternativesList');
        alternativesList.innerHTML = alternatives.map(alt => `
            <div class="alternative-item">
                <span class="alternative-name">${alt.plant} - ${alt.disease}</span>
                <span class="alternative-confidence">${alt.confidence}%</span>
            </div>
        `).join('');
        alternativesSection.style.display = 'block';
    }
    
    // Generate remedies based on disease
    const remedies = getRemedies(top.disease, top.plant);
    const remediesContent = document.getElementById('remediesContent');
    remediesContent.innerHTML = remedies.map(remedy => `
        <div class="remedy-card">
            <i class="fas ${remedy.icon}"></i>
            <strong>${remedy.title}</strong>
            <p>${remedy.description}</p>
        </div>
    `).join('');
    
    // Show results
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getRemedies(disease, plant) {
    const remedies = {
        'healthy': [
            { icon: 'fa-check-circle', title: 'Keep Up the Good Work!', description: `Your ${plant} plant appears healthy. Continue regular watering, ensure adequate sunlight, and monitor for any changes.` },
            { icon: 'fa-calendar-alt', title: 'Preventive Care', description: 'Apply organic fertilizer monthly and check leaves weekly for early signs of issues.' }
        ],
        'scab': [
            { icon: 'fa-tint', title: 'Remove Affected Leaves', description: 'Prune and dispose of infected leaves to prevent spread.' },
            { icon: 'fa-spray-can', title: 'Apply Fungicide', description: 'Use copper-based fungicide or neem oil. Apply every 7-10 days.' }
        ],
        'blight': [
            { icon: 'fa-cut', title: 'Prune Affected Areas', description: 'Remove infected leaves and stems immediately.' },
            { icon: 'fa-tint', title: 'Avoid Overhead Watering', description: 'Water at the base to keep leaves dry.' },
            { icon: 'fa-leaf', title: 'Improve Air Circulation', description: 'Space plants properly and prune for better airflow.' }
        ],
        'rust': [
            { icon: 'fa-spray-can', title: 'Apply Fungicide', description: 'Use sulfur or neem oil-based fungicides.' },
            { icon: 'fa-trash-alt', title: 'Remove Infected Leaves', description: 'Clean up fallen leaves to prevent reinfection.' }
        ],
        'mildew': [
            { icon: 'fa-wind', title: 'Improve Air Flow', description: 'Ensure good air circulation around plants.' },
            { icon: 'fa-spray-can', title: 'Milk Spray', description: 'Mix 1 part milk with 9 parts water and spray on leaves.' }
        ],
        'default': [
            { icon: 'fa-user-md', title: 'Consult an Expert', description: `Contact your local agricultural extension office for specific treatment options.` },
            { icon: 'fa-leaf', title: 'Isolate Plant', description: 'Separate affected plants to prevent spread to healthy ones.' },
            { icon: 'fa-search', title: 'Monitor Closely', description: 'Check plants daily for any changes or progression of symptoms.' }
        ]
    };
    
    // Find matching remedy
    let remedyKey = 'default';
    const diseaseLower = disease.toLowerCase();
    
    if (diseaseLower.includes('healthy')) remedyKey = 'healthy';
    else if (diseaseLower.includes('scab')) remedyKey = 'scab';
    else if (diseaseLower.includes('blight')) remedyKey = 'blight';
    else if (diseaseLower.includes('rust')) remedyKey = 'rust';
    else if (diseaseLower.includes('mildew')) remedyKey = 'mildew';
    
    return remedies[remedyKey] || remedies.default;
}

function showError(message) {
    const resultsSection = document.getElementById('resultsSection');
    const remediesContent = document.getElementById('remediesContent');
    
    resultsSection.style.display = 'block';
    remediesContent.innerHTML = `
        <div class="remedy-card" style="background: #ffebee; border-left-color: #f44336;">
            <i class="fas fa-exclamation-triangle" style="color: #f44336;"></i>
            <strong>Error</strong>
            <p>${message}</p>
        </div>
    `;
    
    document.getElementById('mainResult').innerHTML = '';
    document.getElementById('alternativesSection').style.display = 'none';
}

function resetApp() {
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Reset upload area style
    uploadArea.style.borderColor = '#4caf50';
    uploadArea.style.background = 'transparent';
}

function shareResults() {
    const plant = document.querySelector('.plant-name')?.innerText;
    const disease = document.querySelector('.disease-name')?.innerText;
    const confidence = document.querySelector('.confidence')?.innerText;
    
    if (plant && disease) {
        const text = `🌿 PlantGuard AI Analysis:\nPlant: ${plant}\nDisease: ${disease}\nConfidence: ${confidence}\n\nDetected with AI at PlantGuard!`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Plant Disease Detection Results',
                text: text
            });
        } else {
            navigator.clipboard.writeText(text);
            alert('Results copied to clipboard!');
        }
    }
}

// Check server health
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Server health:', data);
        
        if (!data.model_loaded) {
            console.warn('Model not loaded. Please train the model first.');
        }
    } catch (error) {
        console.error('Server not reachable:', error);
    }
}

// Initialize
initPlantsGrid();
checkHealth();

// Auto-refresh health every 30 seconds
setInterval(checkHealth, 30000);
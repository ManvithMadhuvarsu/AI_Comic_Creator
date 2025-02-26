# MarketLens: AI-Based Product Data Analysis Tool

## Abstract
This project transforms the Product Data Analysis Tool into an AI-driven web application. The tool will act as an intelligent agent that automatically processes uploaded datasets (CSV/JSON), extracts key features, and dynamically generates insights, visualizations, and reports.

The AI agent will guide users through the analysis process, detecting missing columns, handling errors, and suggesting the best visualization types based on data patterns. The backend will be powered by Flask (Python), and the front end will use HTML, CSS, and JavaScript for interactive user engagement.

## Key Features of the AI Agent
- **Automated Feature Extraction** – When a user uploads a dataset, the AI agent scans the file, detects missing or required features, and prepares the data for analysis.
- **Intelligent Data Cleaning** – Identifies missing values, incorrect data types, and outliers, offering preprocessing suggestions.
- **Smart Insights Generation** – Dynamically categorizes data into Market Overview, Customer Behavior, and Product Trends.
- **Adaptive Visualization Recommendations** – Suggests the most relevant charts based on data structure.
- **Automated Report Generation** – Summarizes findings into a detailed PDF report with AI-assisted commentary.

## Technology Stack
- **AI Processing:** Pandas, NumPy, Scikit-learn (for feature detection & data validation)
- **Backend:** Flask (Python) – Manages AI logic, data processing, and report generation
- **Frontend:** HTML, CSS, JavaScript – Provides an interactive user experience
- **Visualization:**
  - **Static:** Matplotlib, Seaborn
  - **Interactive:** Plotly, Dash (AI-driven visualization suggestions)
- **NLP for AI Commentary:** GPT-4 API / BART (for summarizing insights in reports)
- **PDF Generation:** ReportLab

## AI-Driven Webpage Design
1. **Upload Section** – Users upload a dataset (CSV/JSON).
2. **AI Feature Extraction** – The agent automatically scans and extracts relevant columns.
3. **Data Cleaning & Preprocessing** – Handles missing values and incorrect data formats.
4. **Insights & Recommendations** – AI generates key business insights dynamically.
5. **Visualization Dashboard** – Users view AI-suggested static and interactive visualizations.
6. **Export Reports** – Download AI-generated PDF reports with insights and recommendations.

## Activity Flow
1. **User Uploads Dataset** – Upload CSV/JSON file.
2. **AI Agent Analyzes Data** – Detects missing values and errors.
3. **Data Preprocessing** – Cleans and standardizes data.
4. **Feature Extraction** – Identifies key attributes.
5. **Generate Insights** – AI extracts trends and patterns.
6. **Create Visualizations** – Generates static and interactive charts.
7. **Display Results** – Insights and charts shown on the dashboard.
8. **Export Report** – Downloadable PDF with insights.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/MarketLens.git
cd MarketLens

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage
1. Open a web browser and go to `http://127.0.0.1:5000`.
2. Upload a CSV/JSON file.
3. The AI agent will automatically analyze the data and generate insights.
4. View visualizations and download reports.

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries, reach out via [your email] or open an issue on GitHub.


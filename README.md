# Resume Matching Application

This project is an AI-powered Resume Matcher designed to help hiring managers efficiently find the best candidate matches for job openings. It leverages advanced language models and embeddings to analyze resumes and match them against job requirements.

## Features

- **PDF Resume Processing**: Extracts and processes resumes in PDF format.
- **Semantic Embeddings**: Uses Hugging Face models to create semantic embeddings for resumes.
- **Vector Database**: Stores and retrieves resumes using ChromaDB for efficient similarity searches.
- **Conversational AI**: Employs a conversational chain to provide intelligent responses for resume matching queries.
- **Customizable Matching**: Matches resumes based on experience, skills, education, and work history.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: 
  - LangChain
  - Hugging Face
  - ChromaDB
  - Ollama LLM
- **Models**: 
  - `sentence-transformers/all-MiniLM-L6-v2` (Embeddings)
  - `deepseek-r1:7b` (LLM)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the Ollama API is running locally:
   ```bash
   ollama start
   ```

## Configuration

Update the `CONFIG` dictionary in `main.py` to customize paths, models, and parameters:
- `pdf_directory`: Path to the folder containing PDF resumes.
- `db_name`: Name of the vector database.
- `ollama_api`: URL for the Ollama API.
- `model`: LLM model name.
- `embedding_model`: Hugging Face embedding model.

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Enter job details when prompted:
   - Job title
   - Years of experience
   - Required skills
   - Education requirements

3. The application will analyze resumes and provide the most relevant match.

## Example Query

The system generates a query like:
```
You are a hiring manager. I will give you a job description and a resume.
For a [Job Title] position requiring [Experience] years of experience,
with required skills: [Skills], and education: [Education],
you will pick the most relevant resume from the list of resumes.
```

## Results

The application outputs the best-matched resume based on the provided job description.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

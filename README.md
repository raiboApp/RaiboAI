# RaiboAI

RaiboAI is an AI-powered microservice designed for seamless integration with the [RaiboBackend](https://github.com/raiboApp/RaiboBackend/tree/feature/search). It provides intelligent search and content analysis features to enhance the Raibo application ecosystem.

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

`git clone https://github.com/raiboApp/RaiboAI.git`

`cd RaiboAI`

`pip install -r requirements.txt`


### Running the Service

`python app.py`

## Integration with RaiboBackend

RaiboAI is consumed by [RaiboBackend](https://github.com/raiboApp/RaiboBackend/tree/feature/search).  

To integrate:
1. Start the RaiboAI service as shown above.
2. Ensure RaiboBackend is initialised.
3. Use the search and AI features from the Raibo application as usual.

## File Structure

- `app.py` — Main service entry point and API definitions
- `clip_utils.py` — Utility functions for AI processing(feature embeddings functions).
- `requirements.txt` — Python dependencies

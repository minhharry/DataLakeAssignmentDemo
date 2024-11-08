# DataLake Assignment Demo

## Overview

This is a demo project for the DataLake assignment. It contains:
- a data pipeline that extracts keyframes from videos, processes them using OpenCLIP, and saves them to a LanceDB database (vector database).
- a Flask app that allows users to search for images using OpenCLIP and the LanceDB database.

## Setup

### Prerequisites

- Python 3 (Preferably 3.12.6)

### Installation
Clone the repository:

```bash
git clone https://github.com/minhharry/DataLakeAssignmentDemo
cd DataLakeAssignmentDemo
```

Install the required packages using pip:

```bash
pip install -r requirement.txt
```

## Usage

### Data Pipeline

To run the data pipeline, put your videos in the `input` directory and run the following command:

```bash
python data_pipeline.py
```

This will extract keyframes from all videos in the `input` directory and save them to the `output` directory.

### Flask App

To run the Flask app, run the following command:

```bash
python flask-app/main.py
```

This will start the Flask app and you can access it at `http://localhost:5000/`. You can search for images using the OpenCLIP model and the LanceDB database.
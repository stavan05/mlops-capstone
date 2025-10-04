# MLOps Capstone Project

This project demonstrates a complete MLOps pipeline for machine learning model development and deployment.

## Project Structure

```
mlops-capstone/
├── data/
│   └── raw/
│       └── boston_housing.csv          # Raw dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data preprocessing pipeline
│   └── train_model.py                  # Model training pipeline
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # Project documentation
```

## Dataset

The project uses the Boston Housing dataset, which contains information about housing prices in Boston. The dataset includes features like:
- CRIM: Per capita crime rate
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable
- NOX: Nitric oxides concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Distances to five Boston employment centres
- RAD: Index of accessibility to radial highways
- TAX: Full-value property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population
- MEDV: Median value of owner-occupied homes in $1000's (target variable)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops-capstone
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```bash
cd src
python data_preprocessing.py
```

### Model Training
```bash
cd src
python train_model.py
```

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning library
- matplotlib: Plotting library
- seaborn: Statistical data visualization
- dvc: Data version control
- mlflow: ML lifecycle management
- jupyter: Interactive development environment

## MLOps Tools

This project is designed to work with the following MLOps tools:
- **DVC**: For data version control and pipeline management
- **MLflow**: For experiment tracking and model registry
- **Docker**: For containerization
- **Terraform**: For infrastructure as code (deployment step)

## Next Steps

1. Initialize Git repository
2. Set up DVC for data version control
3. Configure MLflow for experiment tracking
4. Create Docker containers
5. Set up CI/CD pipeline
6. Deploy using Terraform

## License

This project is for educational purposes as part of an MLOps capstone project.

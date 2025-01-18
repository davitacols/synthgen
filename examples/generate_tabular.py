from synthgen.core import SynthGen

# Initialize the generator
generator = SynthGen(seed=42)

# Generate a dataset with 100 rows, 3 columns
dataset = generator.generate_tabular(
    rows=100,
    cols=3,
    col_types=['numeric', 'categorical', 'numeric'],
    noise=0.1
)

# Display the first few rows
print(dataset.head())

# Save the dataset to a CSV file
dataset.to_csv('synthetic_dataset.csv', index=False)
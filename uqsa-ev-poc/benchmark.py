import pandas as pd
import os

def run_uqsa_benchmark():
    print("--- 🚀 Generating UQSA-EV Experimental Results ---")
    
    # Using the specific data provided by your experiments
    data = {
        "Intensity": [0, 1000, 2000, 3000, 4000, 5000],
        "Legacy_Latency": [50.0, 52.2, 55.368, 59.93, 133.00, 151.92],
        "UQSA_Latency": [59.0, 67.0, 75.0, 83.0, 91.0, 99.0]
    }

    df = pd.DataFrame(data)
    
    # Save to the current working directory to ensure plotter can find it
    output_file = os.path.join(os.getcwd(), "experiment_results.csv")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created {output_file} successfully.")
    print(df)

if __name__ == "__main__":
    run_uqsa_benchmark()
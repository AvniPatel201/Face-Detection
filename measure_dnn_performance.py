import time
import matplotlib.pyplot as plt
import numpy as np
from Face_detect_DNN import load_and_preprocess_data, create_model
from keras.models import load_model

def measure_performance(model, X_test, batch_sizes=[1, 8, 16, 32]):
    performance_data = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        runtimes = []
        
        # Warm-up run
        model.predict(X_test[:batch_size])
        
        # Measure multiple predictions
        for _ in range(50):  # 50 iterations for each batch size
            start_time = time.time()
            model.predict(X_test[:batch_size])
            end_time = time.time()
            runtimes.append(end_time - start_time)
            
        performance_data[batch_size] = runtimes
    
    return performance_data

def plot_performance(performance_data):
    # Create runtime comparison plot
    plt.figure(figsize=(12, 6))
    plt.boxplot(performance_data.values())
    plt.xlabel('Batch Size')
    plt.ylabel('Runtime (seconds)')
    plt.title('DNN Model Runtime by Batch Size')
    plt.xticks(range(1, len(performance_data) + 1), performance_data.keys())
    plt.grid(True)
    plt.savefig('runtime_comparison.png')
    plt.show()
    
    # Print statistics
    for batch_size, runtimes in performance_data.items():
        print(f"\nBatch size {batch_size}:")
        print(f"Average runtime: {np.mean(runtimes):.4f} seconds")
        print(f"Standard deviation: {np.std(runtimes):.4f} seconds")

def main():
    # Load your data
    base_path = "/Users/avnipatel/Documents/Machine Vision/Final Project/archive-2"
    X_test, _ = load_and_preprocess_data(base_path)
    
    # Load the trained model
    try:
        model = load_model('face_detection_model.keras')
    except:
        print("Trained model not found, creating new model...")
        model = create_model()
    
    # Measure and plot performance
    performance_data = measure_performance(model, X_test)
    plot_performance(performance_data)

if __name__ == "__main__":
    main() 
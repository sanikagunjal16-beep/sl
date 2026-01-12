"""
SIMPLE NAÏVE BAYES CLASSIFICATION - IRIS DATASET DEMO
No pip installations needed! Uses only Python's built-in libraries.
Perfect for academic presentations.
"""
# ============================================
# STEP 0:installibs__python -m pip install numpy pandas matplotlib seaborn scikit-learn
# ============================================
import math
import random
import csv
from collections import defaultdict
import statistics

# ============================================
# STEP 1: Create Iris Dataset (Built-in, no downloads)
# ============================================

print("="*70)
print("ACADEMIC DEMO: NAÏVE BAYES CLASSIFICATION ON IRIS DATASET")
print("="*70)

def create_iris_dataset():
    """
    Create the Iris dataset manually (no downloads needed)
    This follows the original Fisher's Iris dataset structure
    """
    # Iris dataset data (sepal length, sepal width, petal length, petal width, species)
    iris_data = [
        # Setosa (class 0)
        [5.1, 3.5, 1.4, 0.2, 0],
        [4.9, 3.0, 1.4, 0.2, 0],
        [4.7, 3.2, 1.3, 0.2, 0],
        [4.6, 3.1, 1.5, 0.2, 0],
        [5.0, 3.6, 1.4, 0.2, 0],
        [5.4, 3.9, 1.7, 0.4, 0],
        [4.6, 3.4, 1.4, 0.3, 0],
        [5.0, 3.4, 1.5, 0.2, 0],
        [4.4, 2.9, 1.4, 0.2, 0],
        [4.9, 3.1, 1.5, 0.1, 0],
        [5.4, 3.7, 1.5, 0.2, 0],
        [4.8, 3.4, 1.6, 0.2, 0],
        [4.8, 3.0, 1.4, 0.1, 0],
        [4.3, 3.0, 1.1, 0.1, 0],
        [5.8, 4.0, 1.2, 0.2, 0],
        
        # Versicolor (class 1)
        [7.0, 3.2, 4.7, 1.4, 1],
        [6.4, 3.2, 4.5, 1.5, 1],
        [6.9, 3.1, 4.9, 1.5, 1],
        [5.5, 2.3, 4.0, 1.3, 1],
        [6.5, 2.8, 4.6, 1.5, 1],
        [5.7, 2.8, 4.5, 1.3, 1],
        [6.3, 3.3, 4.7, 1.6, 1],
        [4.9, 2.4, 3.3, 1.0, 1],
        [6.6, 2.9, 4.6, 1.3, 1],
        [5.2, 2.7, 3.9, 1.4, 1],
        [5.0, 2.0, 3.5, 1.0, 1],
        [5.9, 3.0, 4.2, 1.5, 1],
        [6.0, 2.2, 4.0, 1.0, 1],
        [6.1, 2.9, 4.7, 1.4, 1],
        [5.6, 2.9, 3.6, 1.3, 1],
        
        # Virginica (class 2)
        [6.3, 3.3, 6.0, 2.5, 2],
        [5.8, 2.7, 5.1, 1.9, 2],
        [7.1, 3.0, 5.9, 2.1, 2],
        [6.3, 2.9, 5.6, 1.8, 2],
        [6.5, 3.0, 5.8, 2.2, 2],
        [7.6, 3.0, 6.6, 2.1, 2],
        [4.9, 2.5, 4.5, 1.7, 2],
        [7.3, 2.9, 6.3, 1.8, 2],
        [6.7, 2.5, 5.8, 1.8, 2],
        [7.2, 3.6, 6.1, 2.5, 2],
        [6.5, 3.2, 5.1, 2.0, 2],
        [6.4, 2.7, 5.3, 1.9, 2],
        [6.8, 3.0, 5.5, 2.1, 2],
        [5.7, 2.5, 5.0, 2.0, 2],
        [5.8, 2.8, 5.1, 2.4, 2]
    ]
    
    # Feature names and class names
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    return iris_data, feature_names, class_names

# Create the dataset
iris_data, feature_names, class_names = create_iris_dataset()

print("\n" + "="*70)
print("STEP 1: DATASET LOADED SUCCESSFULLY")
print("="*70)
print(f"Total samples: {len(iris_data)}")
print(f"Number of features: {len(feature_names)}")
print(f"Classes: {list(class_names.values())}")

print("\nFirst 5 samples:")
print("-" * 60)
print(f"{'Features':<40} | {'Class'}")
print("-" * 60)
for i in range(min(5, len(iris_data))):
    features = iris_data[i][:4]
    class_idx = iris_data[i][4]
    print(f"{str(features):<40} | {class_names[class_idx]}")

# ============================================
# STEP 2: Implement Naïve Bayes Classifier
# ============================================

print("\n" + "="*70)
print("STEP 2: IMPLEMENTING NAÏVE BAYES ALGORITHM")
print("="*70)

class SimpleNaiveBayes:
    """
    Simple Naïve Bayes Classifier from scratch
    Uses Gaussian (Normal) distribution for continuous features
    """
    
    def __init__(self):
        self.class_summaries = {}  # Store mean and std dev for each class
        self.class_priors = {}     # Prior probabilities for each class
    
    def calculate_statistics(self, data):
        """
        Calculate mean and standard deviation for each feature
        """
        if len(data) == 0:
            return [(0, 0)] * len(data[0]) if data else []
        
        # Transpose data to get features in columns
        features = list(zip(*data))
        
        stats = []
        for feature in features:
            mean = sum(feature) / len(feature)
            variance = sum((x - mean) ** 2 for x in feature) / len(feature)
            std_dev = math.sqrt(variance)
            stats.append((mean, std_dev))
        
        return stats
    
    def gaussian_probability(self, x, mean, std_dev):
        """
        Calculate Gaussian probability density function
        Formula: (1 / (sqrt(2π) * σ)) * e^(-(x-μ)²/(2σ²))
        """
        if std_dev == 0:
            # If std_dev is 0, return 1 if x equals mean, else a very small number
            return 1.0 if x == mean else 1e-10
        
        exponent = math.exp(-((x - mean) ** 2) / (2 * (std_dev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent
    
    def train(self, training_data, training_labels):
        """
        Train the Naïve Bayes classifier
        """
        # Separate data by class
        separated_data = {}
        for features, label in zip(training_data, training_labels):
            if label not in separated_data:
                separated_data[label] = []
            separated_data[label].append(features)
        
        # Calculate statistics and priors for each class
        for class_label, class_data in separated_data.items():
            self.class_summaries[class_label] = self.calculate_statistics(class_data)
            self.class_priors[class_label] = len(class_data) / len(training_data)
        
        print("\nTraining completed!")
        print(f"Classes learned: {list(self.class_summaries.keys())}")
        print(f"Prior probabilities: {self.class_priors}")
    
    def predict_single(self, features):
        """
        Predict class for a single sample
        """
        probabilities = {}
        
        for class_label, class_summary in self.class_summaries.items():
            # Start with prior probability (using log to avoid underflow)
            probabilities[class_label] = math.log(self.class_priors[class_label])
            
            # Multiply likelihoods (add logs)
            for i in range(len(features)):
                mean, std_dev = class_summary[i]
                likelihood = self.gaussian_probability(features[i], mean, std_dev)
                # Use log to avoid very small numbers
                if likelihood > 0:
                    probabilities[class_label] += math.log(likelihood)
                else:
                    probabilities[class_label] += math.log(1e-10)  # Very small number
        
        # Return class with highest probability
        return max(probabilities, key=probabilities.get)
    
    def predict(self, test_data):
        """
        Predict classes for multiple samples
        """
        return [self.predict_single(features) for features in test_data]

# ============================================
# STEP 3: Split Data into Train and Test Sets
# ============================================

print("\n" + "="*70)
print("STEP 3: SPLITTING DATA (80% Train, 20% Test)")
print("="*70)

# Separate features and labels
features = [sample[:4] for sample in iris_data]
labels = [sample[4] for sample in iris_data]

# Shuffle the data
combined = list(zip(features, labels))
random.seed(42)  # For reproducible results
random.shuffle(combined)
features, labels = zip(*combined)

# Split into train and test
split_point = int(0.8 * len(features))
train_features = features[:split_point]
train_labels = labels[:split_point]
test_features = features[split_point:]
test_labels = labels[split_point:]

print(f"Training set: {len(train_features)} samples")
print(f"Test set: {len(test_features)} samples")

print("\nClass distribution in training set:")
for class_idx in range(3):
    count = train_labels.count(class_idx)
    print(f"  {class_names[class_idx]}: {count} samples ({count/len(train_labels)*100:.1f}%)")

# ============================================
# STEP 4: Train the Naïve Bayes Classifier
# ============================================

print("\n" + "="*70)
print("STEP 4: TRAINING NAÏVE BAYES CLASSIFIER")
print("="*70)

# Create and train the classifier
nb_classifier = SimpleNaiveBayes()
nb_classifier.train(train_features, train_labels)

# Show learned parameters
print("\nLearned parameters for each class:")
for class_idx in range(3):
    if class_idx in nb_classifier.class_summaries:
        print(f"\n{class_names[class_idx]}:")
        for i, (mean, std) in enumerate(nb_classifier.class_summaries[class_idx]):
            print(f"  {feature_names[i]}: Mean={mean:.3f}, Std={std:.3f}")

# ============================================
# STEP 5: Make Predictions
# ============================================

print("\n" + "="*70)
print("STEP 5: MAKING PREDICTIONS")
print("="*70)

# Make predictions
predictions = nb_classifier.predict(test_features)

print(f"\nMade predictions for {len(test_features)} test samples")

# Display first 5 predictions
print("\nSample predictions (first 5 test samples):")
print("-" * 70)
print(f"{'Actual':<15} {'Predicted':<15} {'Correct?':<10}")
print("-" * 70)
correct_count = 0
for i in range(min(5, len(test_features))):
    actual = class_names[test_labels[i]]
    predicted = class_names[predictions[i]]
    is_correct = actual == predicted
    if is_correct:
        correct_count += 1
    print(f"{actual:<15} {predicted:<15} {'✓' if is_correct else '✗':<10}")

# ============================================
# STEP 6: Confusion Matrix and Performance Metrics
# ============================================

print("\n" + "="*70)
print("STEP 6: CONFUSION MATRIX AND PERFORMANCE METRICS")
print("="*70)

def create_confusion_matrix(actual, predicted, num_classes=3):
    """
    Create confusion matrix manually
    """
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    
    for a, p in zip(actual, predicted):
        cm[a][p] += 1
    
    return cm

# Create confusion matrix
confusion_matrix = create_confusion_matrix(test_labels, predictions)

print("\nCONFUSION MATRIX:")
print("Rows = Actual Class, Columns = Predicted Class")
print("-" * 60)

# Print header
print(f"{'':<15}", end="")
for class_name in class_names.values():
    print(f"{class_name[:8]:>10}", end="")
print(f"{'Total':>10}")

# Print matrix with row totals
for i, class_name in enumerate(class_names.values()):
    print(f"{class_name:<15}", end="")
    row_total = 0
    for j in range(len(class_names)):
        print(f"{confusion_matrix[i][j]:>10}", end="")
        row_total += confusion_matrix[i][j]
    print(f"{row_total:>10}")

# Column totals
print(f"{'Total':<15}", end="")
col_totals = [0] * len(class_names)
for j in range(len(class_names)):
    col_total = sum(confusion_matrix[i][j] for i in range(len(class_names)))
    col_totals[j] = col_total
    print(f"{col_total:>10}", end="")
print(f"{len(test_labels):>10}")

def calculate_metrics_for_class(confusion_matrix, class_index):
    """
    Calculate TP, FP, TN, FN for a specific class
    """
    TP = confusion_matrix[class_index][class_index]
    
    # FP = Sum of column - TP
    FP = sum(confusion_matrix[i][class_index] for i in range(len(confusion_matrix))) - TP
    
    # FN = Sum of row - TP
    FN = sum(confusion_matrix[class_index][j] for j in range(len(confusion_matrix))) - TP
    
    # TN = All other correct predictions
    TN = 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            if i != class_index and j != class_index:
                TN += confusion_matrix[i][j]
    
    return TP, FP, TN, FN

print("\n" + "-" * 70)
print("CLASS-WISE PERFORMANCE METRICS")
print("-" * 70)

all_metrics = []
for class_idx, class_name in class_names.items():
    TP, FP, TN, FN = calculate_metrics_for_class(confusion_matrix, class_idx)
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    error_rate = 1 - accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    all_metrics.append({
        'class': class_name,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'precision': precision,
        'recall': recall
    })
    
    print(f"\n{class_name}:")
    print(f"  TP: {TP:2d}  FP: {FP:2d}  TN: {TN:2d}  FN: {FN:2d}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Error Rate: {error_rate:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")

# ============================================
# STEP 7: Overall Performance Metrics
# ============================================

print("\n" + "="*70)
print("STEP 7: OVERALL MODEL PERFORMANCE")
print("="*70)

# Calculate overall accuracy
correct_predictions = sum(1 for a, p in zip(test_labels, predictions) if a == p)
overall_accuracy = correct_predictions / len(test_labels)
overall_error_rate = 1 - overall_accuracy

# Calculate macro-averaged precision and recall
macro_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
macro_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)

print(f"\nOverall Accuracy:  {overall_accuracy:.4f} ({correct_predictions}/{len(test_labels)} correct)")
print(f"Overall Error Rate: {overall_error_rate:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")

# ============================================
# STEP 8: Simple Visualization (ASCII Art)
# ============================================

print("\n" + "="*70)
print("STEP 8: VISUALIZATION (ASCII CHARTS)")
print("="*70)

def create_ascii_bar_chart(value, max_width=40):
    """Create simple ASCII bar chart"""
    bar_length = int(value * max_width)
    return "█" * bar_length + " " * (max_width - bar_length)

print("\nPERFORMANCE METRICS VISUALIZATION:")
print("-" * 60)

# Overall accuracy bar
print(f"\nOverall Accuracy: {overall_accuracy:.1%}")
print(f"[{create_ascii_bar_chart(overall_accuracy)}]")

# Class-wise accuracy bars
print("\nClass-wise Accuracy:")
for metrics in all_metrics:
    bar = create_ascii_bar_chart(metrics['accuracy'], 30)
    print(f"{metrics['class']:12s} [{bar}] {metrics['accuracy']:.1%}")

# Confusion matrix visualization
print("\nConfusion Matrix Visualization:")
print("-" * 60)
max_count = max(max(row) for row in confusion_matrix)

for i in range(len(confusion_matrix)):
    print(f"\n{class_names[i]:12s}", end="")
    for j in range(len(confusion_matrix)):
        normalized = confusion_matrix[i][j] / max_count if max_count > 0 else 0
        bar_length = int(normalized * 15)
        bar = "█" * bar_length
        print(f" {bar:15s} ({confusion_matrix[i][j]:2d})", end="")
    print()

# ============================================
# STEP 9: Save Results to File
# ============================================

print("\n" + "="*70)
print("STEP 9: SAVING RESULTS TO FILE")
print("="*70)

# Save predictions to a CSV file
with open('naive_bayes_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 
                     'Actual Class', 'Predicted Class', 'Correct'])
    
    # Write data
    for i in range(len(test_features)):
        features = test_features[i]
        actual = class_names[test_labels[i]]
        predicted = class_names[predictions[i]]
        correct = "Yes" if actual == predicted else "No"
        
        row = features + [actual, predicted, correct]
        writer.writerow(row)

print(f"Results saved to 'naive_bayes_results.csv'")

# Save summary to text file
with open('naive_bayes_summary.txt', 'w') as f:
    f.write("NAÏVE BAYES CLASSIFICATION - IRIS DATASET\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Overall Error Rate: {overall_error_rate:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n\n")
    
    f.write("Confusion Matrix:\n")
    for i in range(len(confusion_matrix)):
        f.write(f"{class_names[i]:12s}")
        for j in range(len(confusion_matrix)):
            f.write(f"{confusion_matrix[i][j]:6d}")
        f.write("\n")

print(f"Summary saved to 'naive_bayes_summary.txt'")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*70)
print("DEMO COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nSUMMARY:")
print(f"1. Dataset: Iris (45 samples, 3 classes, 4 features)")
print(f"2. Algorithm: Naïve Bayes with Gaussian distribution")
print(f"3. Train-Test Split: 80%-20%")
print(f"4. Overall Accuracy: {overall_accuracy:.2%}")
print(f"5. Files Created:")
print(f"   - naive_bayes_results.csv (detailed predictions)")
print(f"   - naive_bayes_summary.txt (performance summary)")
print("\nReady for academic presentation!")

# Additional explanation for presentation
print("\n" + "="*70)
print("KEY FORMULAS USED:")
print("="*70)
print("1. Gaussian Probability Density:")
print("   P(x|class) = (1/(√(2π)σ)) * e^(-(x-μ)²/(2σ²))")
print("\n2. Naïve Bayes Classifier:")
print("   P(class|features) ∝ P(class) * Π P(feature_i|class)")
print("\n3. Using Log Probabilities (to avoid underflow):")
print("   log(P(class|features)) = log(P(class)) + Σ log(P(feature_i|class))")

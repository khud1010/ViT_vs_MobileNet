import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_metrics(matrix):
    TP, FP, FN, TN = matrix
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0  # Negative Predictive Value
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0  # False Positive Rate
    fnr = FN / (FN + TP) if (FN + TP) != 0 else 0  # False Negative Rate

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1 Score": f1_score,
        "Negative Predictive Value (NPV)": npv,
        "False Positive Rate (FPR)": fpr,
        "False Negative Rate (FNR)": fnr,
    }

def plot_confusion_matrix(matrix):
    # Reshape the array into a 2x2 matrix
    cm_array = np.array([[matrix[0], matrix[2]],
                         [matrix[1], matrix[3]]])
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Finding', 'Healthy'], 
                yticklabels=['Finding', 'Healthy'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# From mobilenet model training images [TP, FP, FN, TN]
mobile_net_cm = [14071, 6002, 1664, 3859]

# Calculate metrics
mobile_net_metrics = confusion_matrix_metrics(mobile_net_cm)

print('='*50)
print('MobileNet\n')
print('='*50)
# Display the metrics
for metric, value in mobile_net_metrics.items():
    print(f"{metric}: {value:.4f}")
plot_confusion_matrix(mobile_net_cm)


# From vit model training images [TP, FP, FN, TN]
vit_cm = [8865,3872,6870,5989]

# Calculate metrics
vit_metrics = confusion_matrix_metrics(vit_cm)

print('='*50)
print('Vision Transformer\n')
print('='*50)
# Display the metrics
for metric, value in vit_metrics.items():
    print(f"{metric}: {value:.4f}")
plot_confusion_matrix(vit_cm)

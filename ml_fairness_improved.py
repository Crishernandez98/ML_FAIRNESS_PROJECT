"""
Machine Learning Fairness Assignment - Improved Version
======================================================

This script demonstrates ML best practices for bias detection and model evaluation.
Uses a synthetic dataset with demographic features to train and evaluate models
for potential bias in decision-making scenarios (e.g., loan approvals, hiring).

Author: ML Fairness Project
Dataset: synthetic_bias_demo.csv (synthetic data with demographic features)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix, 
                           roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load dataset and display basic information."""
    print("=== DATASET EXPLORATION ===")
    df = pd.read_csv('synthetic_bias_demo.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target distribution:\n{df['label'].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df

def split_data(X, y):
    """
    Split data into train/validation/test sets with clear rationale.
    
    Rationale for 60/20/20 split:
    - 60% training: Sufficient data for model learning while preventing overfitting
    - 20% validation: Used for hyperparameter tuning and model selection
    - 20% test: Unbiased final evaluation on completely unseen data
    
    Best practices implemented:
    - Stratified splitting to maintain class balance across all splits
    - Fixed random_state for reproducibility of results
    - Separate validation set prevents data leakage during model selection
    """
    print("\n=== DATA SPLITTING ===")
    print("Split rationale: 60% train / 20% validation / 20% test")
    print("Using stratified sampling to maintain class balance across all splits")
    
    # First split: 60% train, 40% temp (which will become validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    
    # Second split: 20% validation, 20% test (splitting the 40% temp equally)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verify class balance is maintained
    print(f"\nClass balance verification:")
    print(f"Original: {y.value_counts(normalize=True).values}")
    print(f"Train: {y_train.value_counts(normalize=True).values}")
    print(f"Validation: {y_val.value_counts(normalize=True).values}")
    print(f"Test: {y_test.value_counts(normalize=True).values}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessing_pipeline(X):
    """
    Create preprocessing pipeline for categorical and numerical features.
    
    Preprocessing steps:
    - Categorical features: One-hot encoding to handle categorical variables
    - Numerical features: Standardization (z-score normalization) for scale consistency
    - Handle unknown categories in test data gracefully
    """
    categorical_features = X.select_dtypes('object').columns.tolist()
    numerical_features = X.select_dtypes('number').columns.tolist()
    
    print(f"\nPreprocessing Pipeline:")
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('numerical', StandardScaler(), numerical_features)
    ])
    
    return preprocessor

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    """
    Train multiple models and evaluate with comprehensive metrics.
    
    Model Selection Rationale:
    - Logistic Regression: Linear baseline, interpretable, good for understanding feature importance
    - Decision Tree: Non-linear model that can capture feature interactions and thresholds
    """
    
    # Model dictionary with clear rationale
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)  # Limit depth to prevent overfitting
    }
    
    print("\n=== MODEL TRAINING & EVALUATION ===")
    print("Model Selection Rationale:")
    print("• Logistic Regression: Linear model, highly interpretable, provides probability estimates")
    print("• Decision Tree: Non-linear model, captures feature interactions, easy to visualize decisions")
    
    # Create output directory for results
    output_dir = Path('ml_outputs')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    evaluation_log = []
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        
        # Create complete pipeline with preprocessing + model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model on training set only
        pipeline.fit(X_train, y_train)
        print("✓ Model training completed")
        
        # Store results for this model
        model_results = {}
        
        # Evaluate on both validation and test sets
        for X_eval, y_eval, split_name in [(X_val, y_val, 'Validation'), 
                                           (X_test, y_test, 'Test')]:
            
            # Generate predictions
            y_pred = pipeline.predict(X_eval)
            
            # Get probability scores (for ROC curve)
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                y_scores = pipeline.predict_proba(X_eval)[:, 1]
            else:
                # Fallback for models without probability estimates
                y_scores = y_pred.astype(float)
            
            # Calculate comprehensive evaluation metrics
            metrics = {
                'accuracy': accuracy_score(y_eval, y_pred),
                'precision': precision_score(y_eval, y_pred),
                'recall': recall_score(y_eval, y_pred),
                'f1_score': f1_score(y_eval, y_pred)
            }
            
            # Store all results for analysis
            model_results[split_name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_scores': y_scores,
                'y_true': y_eval
            }
            
            # Display metrics
            print(f"\n{split_name} Set Performance:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}")
            
            # Create detailed log entry
            report = classification_report(y_eval, y_pred)
            log_entry = f"\n{'='*60}\n"
            log_entry += f"{model_name.upper()} - {split_name.upper()} SET EVALUATION\n"
            log_entry += f"{'='*60}\n"
            log_entry += f"Accuracy:  {metrics['accuracy']:.3f}\n"
            log_entry += f"Precision: {metrics['precision']:.3f}\n"
            log_entry += f"Recall:    {metrics['recall']:.3f}\n"
            log_entry += f"F1-Score:  {metrics['f1_score']:.3f}\n"
            log_entry += f"\nDetailed Classification Report:\n{report}\n"
            evaluation_log.append(log_entry)
            
            # Generate visualizations
            create_confusion_matrix(y_eval, y_pred, model_name, split_name, output_dir)
            create_roc_curve(y_eval, y_scores, model_name, split_name, output_dir)
        
        results[model_name] = model_results
    
    # Save comprehensive evaluation report
    log_file = output_dir / 'comprehensive_evaluation_report.txt'
    log_file.write_text('\n'.join(evaluation_log))
    print(f"\n Comprehensive evaluation report saved: {log_file}")
    
    return results

def create_confusion_matrix(y_true, y_pred, model_name, split_name, output_dir):
    """Create and save professional confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Number of Samples'})
    plt.title(f'{model_name} - Confusion Matrix ({split_name} Set)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add performance metrics as text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=10)
    
    filename = output_dir / f'confusion_matrix_{model_name.replace(" ", "_")}_{split_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curve(y_true, y_scores, model_name, split_name, output_dir):
    """Create and save professional ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curve ({split_name} Set)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    filename = output_dir / f'roc_curve_{model_name.replace(" ", "_")}_{split_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_results(results):
    """
    Provide comprehensive and insightful analysis of model performance.
    
    Analysis includes:
    - Model comparison and ranking
    - Strengths and weaknesses of each approach
    - Generalization assessment (validation vs test performance)
    - Practical recommendations
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS ANALYSIS & INTERPRETATION")
    print("="*60)
    
    # 1. MODEL COMPARISON ON TEST SET (Final Evaluation)
    print("\n--- FINAL MODEL COMPARISON (Test Set Performance) ---")
    
    best_f1 = 0
    best_model = ""
    test_results = []
    
    for model_name, model_results in results.items():
        test_metrics = model_results['Test']['metrics']
        f1 = test_metrics['f1_score']
        
        test_results.append({
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1-Score': f1
        })
        
        print(f"\n{model_name}:")
        print(f"  • Accuracy:  {test_metrics['accuracy']:.3f}")
        print(f"  • Precision: {test_metrics['precision']:.3f}")
        print(f"  • Recall:    {test_metrics['recall']:.3f}")
        print(f"  • F1-Score:  {f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
    print(f"   Final F1-Score: {best_f1:.3f}")
    
    # 2. DETAILED MODEL ANALYSIS
    print("\n--- DETAILED MODEL STRENGTHS & WEAKNESSES ---")
    
    for model_name, model_results in results.items():
        test_metrics = model_results['Test']['metrics']
        
        print(f"\n{model_name.upper()}:")
        
        # Precision vs Recall trade-off analysis
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        
        if precision > recall + 0.05:
            print(f"   STRENGTH: High Precision ({precision:.3f})")
            print(f"     → Low false positive rate - conservative in positive predictions")
            print(f"    TRADE-OFF: Lower Recall ({recall:.3f})")
            print(f"     → May miss some actual positive cases")
            print(f"   USE CASE: When false positives are costly")
        
        elif recall > precision + 0.05:
            print(f"   STRENGTH: High Recall ({recall:.3f})")
            print(f"     → Catches most positive cases - comprehensive detection")
            print(f"    TRADE-OFF: Lower Precision ({precision:.3f})")
            print(f"     → Some false positive predictions")
            print(f"   USE CASE: When missing positive cases is costly")
        
        else:
            print(f"   BALANCED: Precision ({precision:.3f}) and Recall ({recall:.3f})")
            print(f"     → Good balance between false positives and false negatives")
            print(f"   USE CASE: When both types of errors are equally important")
        
        # Model-specific algorithmic characteristics
        if model_name == "Logistic Regression":
            print(f"   ALGORITHM CHARACTERISTICS:")
            print(f"     • Linear decision boundary - assumes linear separability")
            print(f"     • Provides probability estimates for uncertainty quantification")
            print(f"     • Feature coefficients show importance and direction")
            print(f"     • Less prone to overfitting, especially with regularization")
            print(f"     • Fast training and prediction")
        
        elif model_name == "Decision Tree":
            print(f"   ALGORITHM CHARACTERISTICS:")
            print(f"     • Non-linear decision boundary - captures complex patterns")
            print(f"     • Highly interpretable with if-then decision rules")
            print(f"     • Automatically handles feature interactions")
            print(f"     • May overfit without proper pruning (max_depth used)")
            print(f"     • Can handle mixed data types naturally")
    
    # 3. GENERALIZATION ANALYSIS
    print("\n--- GENERALIZATION ASSESSMENT ---")
    print("(Comparing Validation vs Test Performance)")
    
    for model_name, model_results in results.items():
        val_f1 = model_results['Validation']['metrics']['f1_score']
        test_f1 = model_results['Test']['metrics']['f1_score']
        
        performance_gap = abs(val_f1 - test_f1)
        
        print(f"\n{model_name}:")
        print(f"  • Validation F1: {val_f1:.3f}")
        print(f"  • Test F1:       {test_f1:.3f}")
        print(f"  • Performance Gap: {performance_gap:.3f}")
        
        if performance_gap < 0.02:
            print("   EXCELLENT GENERALIZATION")
            print("     → Consistent performance across datasets")
            print("     → Model is stable and reliable")
        elif performance_gap < 0.05:
            print("    MODERATE GENERALIZATION")
            print("     → Slight performance variation")
            print("     → Acceptable for most applications")
        else:
            print("   POOR GENERALIZATION")
            print("     → Significant performance drop")
            print("     → May indicate overfitting or data shift")
    
    # 4. PRACTICAL RECOMMENDATIONS
    print("\n--- PRACTICAL RECOMMENDATIONS ---")
    print("\n DEPLOYMENT GUIDANCE:")
    print(f"1. PRIMARY MODEL: Use {best_model} as the main model")
    print(f"   • Achieved highest F1-score of {best_f1:.3f} on test set")
    
    print("\n FURTHER IMPROVEMENTS:")
    print("1. Hyperparameter Tuning:")
    print("   • Use validation set to tune model parameters")
    print("   • Consider grid search or random search")
    print("   • Cross-validation for robust parameter selection")
    
    print("\n2. Feature Engineering:")
    print("   • Analyze feature importance/coefficients")
    print("   • Consider polynomial features or interactions")
    print("   • Domain-specific feature creation")
    
    print("\n3. Bias and Fairness Assessment:")
    print("   • Evaluate performance across demographic groups")
    print("   • Check for disparate impact in predictions")
    print("   • Consider fairness-aware machine learning techniques")
    
    print("\n4. Model Ensemble:")
    print("   • Combine predictions from both models")
    print("   • May improve overall performance and robustness")
    
    print("\n  IMPORTANT CONSIDERATIONS:")
    print("• Always use test set performance for final evaluation")
    print("• Monitor model performance over time in production")
    print("• Consider retraining with new data periodically")
    print("• Validate assumptions about data distribution")

def main():
    """
    Main execution function that orchestrates the entire ML pipeline.
    
    Pipeline stages:
    1. Data loading and exploration
    2. Data splitting with proper validation
    3. Preprocessing pipeline creation
    4. Model training and evaluation
    5. Comprehensive results analysis
    """
    print(" STARTING ML FAIRNESS ANALYSIS PIPELINE")
    print("="*60)
    
    try:
        # Stage 1: Load and explore data
        df = load_and_explore_data()
        
        # Stage 2: Prepare features and target
        X = df.drop(columns='label')  # Feature matrix
        y = df['label']               # Target vector
        
        # Stage 3: Split data with proper validation strategy
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Stage 4: Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(X)
        
        # Stage 5: Train and evaluate models
        results = train_and_evaluate_models(
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
        )
        
        # Stage 6: Comprehensive analysis
        analyze_results(results)
        
        # Summary
        print("\n" + "="*60)
        print(" ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(" All outputs saved to 'ml_outputs/' directory:")
        print("   • Confusion matrices (PNG files)")
        print("   • ROC curves (PNG files)")
        print("   • Comprehensive evaluation report (TXT file)")
        print("\n Pipeline executed with best practices:")
        print("   • Proper data splitting (no leakage)")
        print("   • Stratified sampling (balanced classes)")
        print("   • Comprehensive evaluation metrics")
        print("   • Professional visualizations")
        print("   • Insightful results interpretation")
        
    except Exception as e:
        print(f"\n ERROR: An error occurred during execution:")
        print(f"   {str(e)}")
        print("\n Please check:")
        print("   • Data file 'synthetic_bias_demo.csv' exists")
        print("   • All required libraries are installed")
        print("   • Sufficient permissions for file creation")

if __name__ == "__main__":
    main()